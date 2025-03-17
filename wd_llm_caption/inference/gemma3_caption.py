import time
from argparse import Namespace
from pathlib import Path

from PIL import Image

from . import llm_inference, get_llm_dtype
from ..utils.image_process_util import image_process, image_process_image, encode_image_to_base64
from ..utils.logger_util import Logger


class Gemma3:
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
            from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
        except ImportError as ie:
            self.logger.error(f'Import transformers Failed!\nDetails: {ie}')
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
        # Load Gemma3 model
        self.llm = Gemma3ForConditionalGeneration.from_pretrained(self.llm_path,
                                                        device_map="auto" if not self.args.llm_use_cpu else "cpu",
                                                        torch_dtype=llm_dtype,
                                                        quantization_config=qnt_config)
        self.llm.eval()
        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')
        # Load processor for `Gemma3`
        start_time = time.monotonic()
        self.logger.info(f'Loading processor with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        self.llm_processor = AutoProcessor.from_pretrained(self.llm_path, use_fast=False)
        self.logger.info(f'Processor Loaded in {time.monotonic() - start_time:.1f}s.')

    def get_caption(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0,
            top_p: float = 0,
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
            image = encode_image_to_base64(image)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            self.logger.debug(f"\nChat_template:\n{messages}")
            inputs = self.llm_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.llm.device, dtype=self.llm.dtype)
            input_len = inputs["input_ids"].shape[-1]
            # Generate caption
            if temperature == 0:
                temperature = 1.0
                self.logger.warning(f'LLM temperature not set, using default value {temperature}')
            else:
                self.logger.debug(f'LLM temperature is {temperature}')
            if top_p == 0:
                top_p = 0.95
                self.logger.warning(f'LLM top_p not set, using default value {top_p}')
            else:
                self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
            if max_new_tokens == 0:
                max_new_tokens = 512
                self.logger.warning(f'LLM max_new_tokens not set, using default value {max_new_tokens}')
            else:
                self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
            with torch.inference_mode():
                generation = self.llm.generate(**inputs, temperature=temperature,
                                               min_p=0.00, top_k=64, top_p=top_p,
                                               max_new_tokens=max_new_tokens, do_sample=True)
                generation = generation[0][input_len:]
           
            content = self.llm_processor.decode(generation, skip_special_tokens=True)
            content = content.rstrip("<end_of_turn>")
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
