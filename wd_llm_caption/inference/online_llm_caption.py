import os
import time
from argparse import Namespace
from pathlib import Path

from PIL import Image
from openai import OpenAI

from . import llm_inference
from ..utils.image_process_util import encode_image_to_base64, image_process, image_process_image
from ..utils.logger_util import Logger


class OpenAIapi:
    def __init__(
            self,
            logger: Logger,
            args: Namespace,
    ):
        self.logger = logger
        self.args = args

        self.client = None

        if not args.llm_config:
            llm_config_file = os.path.join(Path(__file__).parent.parent, 'configs', 'default_joy.json')

    def load_model(self):
        # Load VolcengineArk model
        start_time = time.monotonic()
        self.logger.info(f'Setting Online LLM `{self.args.llm_model_name}` Config...')
        if not self.args.llm_online_api_key:
            self.logger.error("llm_online_api_key not set!!!")
            raise ValueError
        if not self.args.llm_online_base_url:
            self.logger.error("llm_online_base_url not set!!!")
            raise ValueError
        if not self.args.llm_model_name:
            self.logger.error("model_name not set!!!")
            raise ValueError
        self.logger.info(f"Using Online LLM service from `{self.args.llm_online_base_url}`")
        self.client = OpenAI(
            api_key=self.args.llm_online_api_key,
            base_url=self.args.llm_online_base_url,
        )
        self.logger.info(f'LLM Config set in {time.monotonic() - start_time:.1f}s.')

    def get_caption(
            self,
            image: Image.Image,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0,
            top_p: float = 0,
            max_new_tokens: int = 0
    ) -> str:
        image = image_process(image, target_size=int(self.args.image_size))
        image = image_process_image(image)
        image = encode_image_to_base64(image, image_format="PNG")
        messages = [
            {"role": "system", "content": system_prompt, },
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_prompt, },
                 {"type": "image_url",
                  "image_url": {"url": image, "detail": "high"}
                  }, ],
             }
        ]
        # Set params:
        params = {}
        if temperature == 0:
            self.logger.warning(f'LLM temperature not set, using its default value')
        else:
            self.logger.debug(f'LLM temperature is {temperature}')
            params['temperature'] = temperature
        if top_p == 0:
            self.logger.warning(f'LLM top_p not set, using its default value')
        else:
            self.logger.debug(f'LLM top_p is {max_new_tokens}')
            params['top_p'] = top_p
        if max_new_tokens == 0:
            self.logger.warning(f'LLM max_new_tokens not set, using its default value')
        else:
            self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
            params['max_new_tokens'] = max_new_tokens
        # Send response
        response = self.client.chat.completions.create(
            model=self.args.llm_model_name,
            messages=messages,
            stream=False,
            **params
        )
        # Read response
        response = response.choices[0]
        output = response.message.content
        return output

    def inference(self):
        llm_inference(self)

    def unload_model(self) -> bool:
        llm_unloaded = True
        self.logger.debug(f"Online model don't need unload!")
        return llm_unloaded
