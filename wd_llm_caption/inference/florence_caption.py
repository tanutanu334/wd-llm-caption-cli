import time
from argparse import Namespace
from pathlib import Path

from PIL import Image

from . import llm_inference
from ..utils.image_process_util import image_process, image_process_image
from ..utils.logger_util import Logger


class Florence2:
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
            from transformers import (AutoProcessor, AutoTokenizer, BitsAndBytesConfig, AutoModel, AutoModelForCausalLM,
                                      LlavaForConditionalGeneration, PreTrainedTokenizer, PreTrainedTokenizerFast)
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
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path,  # Load `florence` model
                                                        device_map="cuda" if not self.args.llm_use_cpu else "cpu",
                                                        torch_dtype=llm_dtype, trust_remote_code=True)
        self.llm.eval()
        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')
        # Load processor
        start_time = time.monotonic()
        self.logger.info(f'Loading processor with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        self.llm_processor = AutoProcessor.from_pretrained(self.llm_path, trust_remote_code=True)
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

            if system_prompt or user_prompt:
                self.logger.warning(f"Florence models don't support system prompt or user prompt!")
            if temperature != 0 or max_new_tokens != 0:
                self.logger.warning(f"Florence models don't support temperature or max tokens!")

            def run_inference(task_prompt, image, text_input=None):
                if text_input is None:
                    input_prompt = task_prompt
                else:
                    input_prompt = task_prompt + text_input
                get_inputs = (self.llm_processor(text=input_prompt, images=image, return_tensors="pt").
                              to(device=self.llm.device, dtype=self.llm.dtype))
                generated_ids = self.llm.generate(
                    input_ids=get_inputs["input_ids"],
                    pixel_values=get_inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                generated_text = self.llm_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.llm_processor.post_process_generation(generated_text, task=task_prompt,
                                                                           image_size=(
                                                                               image.width, image.height))
                return parsed_answer[task_prompt]

            content = run_inference("<MORE_DETAILED_CAPTION>", image)
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
