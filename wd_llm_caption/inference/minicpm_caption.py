import time
from argparse import Namespace
from pathlib import Path

from PIL import Image

from . import llm_inference
from ..utils.image_process_util import image_process, image_process_image
from ..utils.logger_util import Logger


class Minicpm2:
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
        self.llm_tokenizer = None
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
            from transformers import AutoProcessor, AutoTokenizer, AutoModel, BitsAndBytesConfig
        except ImportError as ie:
            self.logger.error(f'Import transformers Failed!\nDetails: {ie}')
            raise ImportError

        device = "cpu" if self.args.llm_use_cpu else "cuda"
        # Load LLM
        self.logger.info(
            f'Loading LLM `{self.args.llm_model_name}` with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        start_time = time.monotonic()
        # LLM dType
        llm_dtype = torch.float32 if self.args.llm_use_cpu or self.args.llm_dtype == "fp32" else \
            torch.bfloat16 if self.args.llm_dtype == "bf16" else \
                torch.float16 if self.args.llm_dtype == "fp16" else torch.float32
        self.logger.info(f'LLM dtype: {llm_dtype}')
        # LLM BNB quantization config
        if self.args.llm_qnt == "4bit":
            qnt_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_quant_storage=torch.uint8,
                                            llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
                                            bnb_4bit_compute_dtype=llm_dtype,
                                            bnb_4bit_use_double_quant=True)
            self.logger.info(f'LLM 4bit quantization: Enabled')
        elif self.args.llm_qnt == "8bit":
            qnt_config = BitsAndBytesConfig(load_in_8bit=True,
                                            llm_int8_enable_fp32_cpu_offload=True)
            self.logger.info(f'LLM 8bit quantization: Enabled')
        else:
            qnt_config = None

        self.llm = AutoModel.from_pretrained(self.llm_path, device_map="cuda" if not self.args.llm_use_cpu else "cpu",
                                             torch_dtype=llm_dtype,
                                             quantization_config=qnt_config,
                                             trust_remote_code=True)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.llm.eval()

        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')

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
            device = "cpu" if self.args.llm_use_cpu else "cuda"
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
            self.logger.debug(f'Using user prompt:{user_prompt}')
            messages = [{'role': 'user', 'content': [image, f'{user_prompt}']}]
            if temperature == 0 and max_new_tokens == 0:
                max_new_tokens = 2048
                self.logger.warning(f'LLM temperature and max_new_tokens not set, only '
                                    f'using default max_new_tokens value {max_new_tokens}')
                params = {
                    'num_beams': 3,
                    'repetition_penalty': 1.2,
                    "max_new_tokens": max_new_tokens
                }
            else:
                if temperature == 0:
                    temperature = 0.7
                    self.logger.warning(f'LLM temperature not set, using default value {temperature}')
                else:
                    self.logger.debug(f'LLM temperature is {temperature}')
                if max_new_tokens == 0:
                    max_new_tokens = 2048
                    self.logger.warning(f'LLM max_new_tokens not set, using default value {max_new_tokens}')
                else:
                    self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
                params = {
                    'top_p': 0.8,
                    'top_k': 100,
                    'temperature': temperature,
                    'repetition_penalty': 1.05,
                    "max_new_tokens": max_new_tokens
                }
            params["max_inp_length"] = 4352
            content = self.llm.chat(image=image, msgs=messages, tokenizer=self.llm_tokenizer,
                                    system_prompt=system_prompt if system_prompt else None,
                                    sampling=False, stream=False, **params)

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
            if hasattr(self, "llm_tokenizer"):
                del self.llm_tokenizer
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
