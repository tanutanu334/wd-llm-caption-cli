import time
from argparse import Namespace
from pathlib import Path

from PIL import Image

from . import llm_inference, get_llm_dtype
from ..utils.image_process_util import image_process, image_process_image
from ..utils.logger_util import Logger


class Janus:
    def __init__(
            self,
            logger: Logger,
            models_paths: tuple[Path],
            args: Namespace,
    ):
        self.logger = logger
        self.args = args

        if len(models_paths) != 1:
            self.logger.error(self.logger.error(f"Invalid models paths: {models_paths}!!!"))
            raise ValueError

        self.llm_path = models_paths[0]
        self.llm_processor = None
        self.tokenizer = None
        self.llm = None

    def load_model(self):
        # Import torch
        try:
            import torch
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError
        # Import transformers
        try:
            from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
        except ImportError as ie:
            self.logger.error(f'Import transformers Failed!\nDetails: {ie}')
            raise ImportError
        # Import Janus
        try:
            from ..third_party.Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
            from ..third_party.Janus.janus.utils.io import load_pil_images
        except ImportError as ie:
            self.logger.error(f'Import Janus Failed!\nDetails: {ie}')
            raise ImportError

        # Load LLM
        self.logger.info(
            f'Loading LLM `{self.args.llm_model_name}` with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        start_time = time.monotonic()
        # LLM dType
        llm_dtype = get_llm_dtype(logger=self.logger, args=self.args)
        self.logger.info(f'LLM dtype: {llm_dtype}')
        # LLM BNB quantization config
        if self.args.llm_qnt == "4bit":
            qnt_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=llm_dtype,
                                            bnb_4bit_use_double_quant=True)
            self.logger.info(f'LLM 4bit quantization: Enabled')
        elif self.args.llm_qnt == "8bit":
            qnt_config = BitsAndBytesConfig(load_in_8bit=True,
                                            llm_int8_enable_fp32_cpu_offload=True)
            self.logger.info(f'LLM 8bit quantization: Enabled')
        else:
            qnt_config = None
        # Load Janus model
        llm_config = AutoConfig.from_pretrained(self.llm_path)
        llm_language_config = llm_config.language_config
        llm_language_config._attn_implementation = 'eager'
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path,
                                                        device_map="auto" if not self.args.llm_use_cpu else "cpu",
                                                        torch_dtype=llm_dtype,
                                                        quantization_config=qnt_config,
                                                        language_config=llm_language_config,
                                                        trust_remote_code=True)
        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')
        # Load processor for `Janus`
        start_time = time.monotonic()
        self.logger.info(f'Loading processor with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        self.llm_processor = VLChatProcessor.from_pretrained(self.llm_path)
        self.tokenizer = self.llm_processor.tokenizer
        self.logger.info(f'Processor Loaded in {time.monotonic() - start_time:.1f}s.')

    def get_caption(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0,
            max_new_tokens: int = 0,
    ) -> str:
        # Import torch
        try:
            import torch
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError
        with torch.no_grad():
            # Cleaning VRAM cache
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()

            image = image_process(image, target_size=int(self.args.image_size))
            self.logger.debug(f"Resized image shape: {image.shape}")
            image = image_process_image(image)

            if system_prompt:
                self.logger.warning(f"`{self.args.llm_model_name}` doesn't support system prompt, "
                                    f"adding system prompt into user prompt...")
                user_prompt = system_prompt + "\n" + user_prompt
            messages = [
                {"role": "<|User|>",
                 "content": f"<image_placeholder>\n{user_prompt}",
                 "images": [image], },
                {"role": "<|Assistant|>", "content": ""},
            ]
            self.logger.debug(f"\nChat_template:\n{messages}")
            prepare_inputs = self.llm_processor(conversations=messages, images=[image], force_batchify=True
                                                ).to("cpu" if self.args.llm_use_cpu else "cuda",
                                                     dtype=get_llm_dtype(logger=self.logger, args=self.args))
            inputs_embeds = self.llm.prepare_inputs_embeds(**prepare_inputs)
            # Generate caption
            self.logger.debug(f'LLM temperature is {temperature}')
            self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
            if temperature == 0:
                temperature = 0.1
                self.logger.warning(f'LLM temperature not set, using default value {temperature}')
            else:
                self.logger.debug(f'LLM temperature is {temperature}')
            if max_new_tokens == 0:
                max_new_tokens = 512
                self.logger.warning(f'LLM max_new_tokens not set, using default value {max_new_tokens}')
            else:
                self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
            outputs = self.llm.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False if temperature == 0 else True,
                use_cache=True,
                temperature=temperature,
                top_p=0.95,
            )
            content = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            content_list = str(content).split(".")
            unique_content = list(dict.fromkeys(content_list))
            unique_content = '.'.join(unique_content)
            return unique_content

    def inference(self):
        llm_inference(self)

    def unload_model(self) -> bool:
        llm_unloaded = False
        # Unload LLM
        if hasattr(self, "llm") and self.llm is not None:
            self.logger.info(f'Unloading LLM...')
            start = time.monotonic()
            del self.llm
            del self.llm_processor
            del self.tokenizer
            if hasattr(self, "llm_processer"):
                del self.llm_processor
            self.logger.info(f'LLM unloaded in {time.monotonic() - start:.1f}s.')
            llm_unloaded = True

        try:
            import torch
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError

        return llm_unloaded
