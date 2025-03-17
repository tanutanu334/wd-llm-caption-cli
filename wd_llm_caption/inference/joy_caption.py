import json
import os
import time
from argparse import Namespace
from pathlib import Path

from PIL import Image

from . import llm_inference, get_llm_dtype
from ..utils.image_process_util import image_process, image_process_image
from ..utils.logger_util import Logger


class Joy:
    def __init__(
            self,
            logger: Logger,
            models_paths: tuple[Path],
            args: Namespace,
    ):
        self.logger = logger
        self.args = args

        if (self.args.llm_model_name == "Joy-Caption-Pre-Alpha" and len(models_paths) != 3) or \
                (self.args.llm_model_name in ["Joy-Caption-Alpha-One", "Joy-Caption-Alpha-Two"]
                 and len(models_paths) != 4) or \
                (self.args.llm_model_name == "Joy-Caption-Alpha-Two-Llava" and len(models_paths) != 1):
            self.logger.error(self.logger.error(f"Invalid models paths: {models_paths}!!!"))
            raise ValueError

        if self.args.llm_model_name == "Joy-Caption-Alpha-Two-Llava":
            self.llm_path = models_paths[0]
        else:
            self.image_adapter_path = models_paths[0]
            self.clip_path = models_paths[1]
            self.llm_path = models_paths[2]
            if self.args.llm_model_name in ["Joy-Caption-Alpha-One", "Joy-Caption-Alpha-Two"] and \
                    self.args.llm_patch:
                self.llm_patch_path = models_paths[3]

        self.image_adapter = None
        self.clip_processor = None
        self.clip_model = None
        self.llm_tokenizer = None
        self.llm = None

    def load_model(self):
        # Import torch
        try:
            import torch
            from torch import nn
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError
        # Import transformers
        try:
            from transformers import (AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig,
                                      LlavaForConditionalGeneration, PreTrainedTokenizer, PreTrainedTokenizerFast)
        except ImportError as ie:
            self.logger.error(f'Import transformers Failed!\nDetails: {ie}')
            raise ImportError

        device = "cpu" if self.args.llm_use_cpu else "cuda"
        # Load CLIP model for Joy
        if self.args.llm_model_name != "Joy-Caption-Alpha-Two-Llava":
            self.logger.info(f'Loading CLIP with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
            start_time = time.monotonic()
            self.clip_processor = AutoProcessor.from_pretrained(self.clip_path)
            self.clip_model = AutoModel.from_pretrained(self.clip_path)
            self.clip_model = self.clip_model.vision_model

            if self.args.llm_model_name != "Joy-Caption-Pre-Alpha":
                self.logger.info(f"Loading custom LLM vision model...")
                checkpoint = torch.load(os.path.join(self.image_adapter_path, "clip_model.pt"), map_location='cpu')
                checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
                self.clip_model.load_state_dict(checkpoint)
                del checkpoint

            self.clip_model.eval()
            self.clip_model.requires_grad_(False)
            self.clip_model.to(device)
            self.logger.info(f'CLIP Loaded in {time.monotonic() - start_time:.1f}s.')
        # Load LLM
        self.logger.info(
            f'Loading LLM `{self.args.llm_model_name}` with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
        start_time = time.monotonic()
        # Load tokenizer
        if self.args.llm_model_name in ["Joy-Caption-Pre-Alpha", "Joy-Caption-Alpha-One"]:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path, use_fast=False)
        else:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_patch_path if self.args.llm_model_name == "Joy-Caption-Alpha-Two" else self.llm_path,
                use_fast=True)
        assert (isinstance(self.llm_tokenizer, PreTrainedTokenizer) or
                isinstance(self.llm_tokenizer, PreTrainedTokenizerFast)), \
            f"Tokenizer is of type {type(self.llm_tokenizer)}"
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

        if self.args.llm_model_name == "Joy-Caption-Alpha-Two-Llava":
            # TODO: make Joy-Caption-Alpha-Two-Llava quantization work.
            # if self.args.llm_qnt != "none":
            #     self.logger.warning(f"`Joy-Caption-Alpha-Two-Llava` current not support quantization.")
            #     self.args.llm_qnt = "none"
            #     qnt_config = None
            self.llm = LlavaForConditionalGeneration.from_pretrained(self.llm_path,  # Load `Llava` model
                                                                     device_map="auto" \
                                                                         if not self.args.llm_use_cpu else "cpu",
                                                                     torch_dtype=llm_dtype,
                                                                     quantization_config=qnt_config)
        else:
            # Load `Llama 3.1 Instruct` LoRA patch
            if self.args.llm_model_name in ["Joy-Caption-Alpha-One", "Joy-Caption-Alpha-Two"] and \
                    self.args.llm_patch and self.llm_patch_path:
                adapter_config_json = os.path.join(self.llm_patch_path, "adapter_config.json")
                if os.path.isfile(adapter_config_json):
                    with open(adapter_config_json, 'r') as file:
                        data = json.load(file)
                    if data['base_model_name_or_path'] != str(self.llm_path):
                        self.logger.warning(f"Found `{adapter_config_json}` need to patch, patching...")
                        data['base_model_name_or_path'] = str(self.llm_path)
                        with open(adapter_config_json, 'w') as file:
                            json.dump(data, file, indent=2)
                        self.logger.info(f"`{adapter_config_json}` patched.")
                    else:
                        self.logger.warning(f"`{adapter_config_json}` already patched.")
                    self.llm_path = self.llm_patch_path
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path,
                                                            device_map="auto" if not self.args.llm_use_cpu else "cpu",
                                                            torch_dtype=llm_dtype,
                                                            quantization_config=qnt_config)
        self.llm.eval()
        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')
        # Load Image Adapter for Joy
        if self.args.llm_model_name != "Joy-Caption-Alpha-Two-Llava":
            if self.args.llm_model_name == "Joy-Caption-Pre-Alpha":
                class ImageAdapter(nn.Module):
                    def __init__(self, input_features: int, output_features: int):
                        super().__init__()
                        self.linear1 = nn.Linear(input_features, output_features)
                        self.activation = nn.GELU()
                        self.linear2 = nn.Linear(output_features, output_features)

                    def forward(self, vision_outputs: torch.Tensor):
                        x = self.linear1(vision_outputs)
                        x = self.activation(x)
                        x = self.linear2(x)
                        return x
            else:
                class ImageAdapter(nn.Module):
                    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool,
                                 num_image_tokens: int, deep_extract: bool):
                        super().__init__()
                        self.deep_extract = deep_extract

                        if self.deep_extract:
                            input_features = input_features * 5

                        self.linear1 = nn.Linear(input_features, output_features)
                        self.activation = nn.GELU()
                        self.linear2 = nn.Linear(output_features, output_features)
                        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
                        self.pos_emb = None if not pos_emb else nn.Parameter(
                            torch.zeros(num_image_tokens, input_features))

                        # Mode token
                        # self.mode_token = nn.Embedding(n_modes, output_features)
                        # self.mode_token.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

                        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
                        self.other_tokens = nn.Embedding(3, output_features)
                        self.other_tokens.weight.data.normal_(mean=0.0,
                                                              std=0.02)  # Matches HF's implementation of llama3

                    def forward(self, vision_outputs: torch.Tensor):
                        if self.deep_extract:
                            x = torch.concat((
                                vision_outputs[-2],
                                vision_outputs[3],
                                vision_outputs[7],
                                vision_outputs[13],
                                vision_outputs[20],
                            ), dim=-1)
                            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
                            assert x.shape[-1] == vision_outputs[-2].shape[
                                -1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
                        else:
                            x = vision_outputs[-2]

                        x = self.ln1(x)

                        if self.pos_emb is not None:
                            assert x.shape[
                                   -2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
                            x = x + self.pos_emb

                        x = self.linear1(x)
                        x = self.activation(x)
                        x = self.linear2(x)

                        # Mode token
                        # mode_token = self.mode_token(mode)
                        # assert mode_token.shape == (x.shape[0], mode_token.shape[1], x.shape[2]), f"Expected {(x.shape[0], 1, x.shape[2])}, got {mode_token.shape}"
                        # x = torch.cat((x, mode_token), dim=1)

                        # <|image_start|>, IMAGE, <|image_end|>
                        other_tokens = self.other_tokens(
                            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
                        assert other_tokens.shape == (
                            x.shape[0], 2,
                            x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
                        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

                        return x

                    def get_eot_embedding(self):
                        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

            self.logger.info(f'Loading Image Adapter with {"CPU" if self.args.llm_use_cpu else "GPU"}...')
            start_time = time.monotonic()
            if self.args.llm_model_name == "Joy-Caption-Pre-Alpha":
                self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.llm.config.hidden_size)
            else:
                self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.llm.config.hidden_size,
                                                  False, False, 38, False)
            self.image_adapter.load_state_dict(torch.load(os.path.join(self.image_adapter_path, "image_adapter.pt"),
                                                          map_location="cpu"))
            self.image_adapter.eval()
            self.image_adapter.to(device)
            self.logger.info(f'Image Adapter Loaded in {time.monotonic() - start_time:.1f}s.')

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
            import torch.amp.autocast_mode
            import torchvision.transforms.functional as TVF
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError
        with torch.no_grad():
            device = "cpu" if self.args.llm_use_cpu else "cuda"
            # Cleaning VRAM cache
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()

            # Preprocess image
            self.logger.warning(f"`{self.args.llm_model_name}` force resize input image to 384 pixels!")
            image = image_process(image, target_size=384)
            image = image_process_image(image)
            pixel_values = TVF.pil_to_tensor(image)

            llm_dtype = torch.float32 if self.args.llm_use_cpu or self.args.llm_dtype == "fp32" else \
                torch.bfloat16 if self.args.llm_dtype == "bf16" else \
                    torch.float16 if self.args.llm_dtype == "fp16" else torch.float32
            # Normalize the image
            if self.args.llm_model_name == "Joy-Caption-Alpha-Two-Llava":
                pixel_values = pixel_values / 255.0
                pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
                pixel_values = pixel_values.to(llm_dtype).unsqueeze(0)
            else:
                pixel_values = pixel_values.unsqueeze(0) / 255.0
                pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
                pixel_values = pixel_values.to(device)

            if self.args.llm_model_name == "Joy-Caption-Pre-Alpha":
                # Tokenize the prompt
                self.logger.warning(f"{self.args.llm_model_name} doesn't support system prompt, "
                                    f"adding system prompt into user prompt...")
                self.logger.debug(f'Using user prompt:{user_prompt}')
                user_prompt = system_prompt+"\n"+user_prompt
                prompt = self.llm_tokenizer.encode(user_prompt,
                                                   return_tensors='pt',
                                                   padding=False,
                                                   truncation=False,
                                                   add_special_tokens=False)
                # Embed image
                with torch.amp.autocast_mode.autocast(device, enabled=True):
                    vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                    image_features = vision_outputs.hidden_states[-2]
                    embedded_images = self.image_adapter(image_features)
                    embedded_images = embedded_images.to(device)
                # Embed prompt
                prompt_embeds = self.llm.model.embed_tokens(prompt.to(device))
                assert prompt_embeds.shape == (1, prompt.shape[1],
                                               self.llm.config.hidden_size), \
                    (f"Prompt shape is {prompt_embeds.shape}, "
                     f"expected {(1, prompt.shape[1], self.llm.config.hidden_size)}")
                embedded_bos = self.llm.model.embed_tokens(torch.tensor([[self.llm_tokenizer.bos_token_id]],
                                                                        device=self.llm.device,
                                                                        dtype=torch.int64))
                # Construct prompts
                inputs_embeds = torch.cat([
                    embedded_bos.expand(embedded_images.shape[0], -1, -1),
                    embedded_images.to(dtype=embedded_bos.dtype),
                    prompt_embeds.expand(embedded_images.shape[0], -1, -1),
                ], dim=1)

                input_ids = torch.cat([
                    torch.tensor([[self.llm_tokenizer.bos_token_id]], dtype=torch.long),
                    torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                    prompt,
                ], dim=1).to(device)
                attention_mask = torch.ones_like(input_ids)
                # Generate caption
                if temperature == 0:
                    temperature = 0.5
                    self.logger.warning(f'LLM temperature not set, using default value {temperature}')
                else:
                    self.logger.debug(f'LLM temperature is {temperature}')
                if max_new_tokens == 0:
                    max_new_tokens = 300
                    self.logger.warning(f'LLM max_new_tokens not set, using default value {max_new_tokens}')
                else:
                    self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')
                generate_ids = self.llm.generate(input_ids,
                                                 inputs_embeds=inputs_embeds,
                                                 attention_mask=attention_mask,
                                                 max_new_tokens=max_new_tokens,
                                                 do_sample=True, top_k=10,
                                                 temperature=temperature,
                                                 suppress_tokens=None)
                # Trim off the prompt
                generate_ids = generate_ids[:, input_ids.shape[1]:]
                if generate_ids[0][-1] == self.llm_tokenizer.eos_token_id:
                    generate_ids = generate_ids[:, :-1]

                content = self.llm_tokenizer.batch_decode(generate_ids,
                                                          skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=False)[0]

            else:
                self.logger.debug(f'Using system prompt: {system_prompt}')
                self.logger.debug(f'Using user prompt: {user_prompt}')
                # Build the conversation
                convo = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]

                # Format the conversation
                convo_string = self.llm_tokenizer.apply_chat_template(convo, tokenize=False,
                                                                      add_generation_prompt=True)
                assert isinstance(convo_string, str)

                if temperature == 0:
                    temperature = 0.6
                    self.logger.warning(f'LLM temperature not set, using default value {temperature}')
                else:
                    self.logger.debug(f'LLM temperature is {temperature}')
                if max_new_tokens == 0:
                    max_new_tokens = 300
                    self.logger.warning(f'LLM max_new_tokens not set, using default value {max_new_tokens}')
                else:
                    self.logger.debug(f'LLM max_new_tokens is {max_new_tokens}')

                if self.args.llm_model_name == "Joy-Caption-Alpha-Two-Llava":
                    # Tokenize the conversation
                    # prompt_str is tokenized separately so we can do the calculations below
                    convo_tokens = self.llm_tokenizer.encode(convo_string, add_special_tokens=False,
                                                             truncation=False)
                    # Repeat the image tokens
                    input_tokens = []
                    for token in convo_tokens:
                        if token == self.llm.config.image_token_index:
                            input_tokens.extend(
                                [self.llm.config.image_token_index] * self.llm.config.image_seq_length)
                        else:
                            input_tokens.append(token)

                    input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
                    input_ids = input_ids.to(device)
                    attention_mask = torch.ones_like(input_ids)
                    attention_mask = attention_mask.to(device)
                    # cast pixel_values to quantization's precision
                    if self.args.llm_qnt == "4bit":
                        pixel_values = pixel_values.to(torch.uint8)
                    elif self.args.llm_qnt == "8bit":
                        pixel_values = pixel_values.to(torch.int8)
                    pixel_values = pixel_values.to(device)

                    # Generate the caption
                    generate_ids = \
                        self.llm.generate(input_ids=input_ids, pixel_values=pixel_values,
                                          attention_mask=attention_mask,
                                          temperature=temperature, max_new_tokens=max_new_tokens,
                                          do_sample=True, suppress_tokens=None, use_cache=True)[0]

                    # Trim off the prompt
                    generate_ids = generate_ids[input_ids.shape[1]:]

                    # Decode the caption
                    content = self.llm_tokenizer.decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                else:
                    # Tokenize the conversation
                    # prompt_str is tokenized separately so we can do the calculations below
                    convo_tokens = self.llm_tokenizer.encode(convo_string, return_tensors="pt",
                                                             add_special_tokens=False,
                                                             truncation=False)

                    prompt_tokens = self.llm_tokenizer.encode(user_prompt, return_tensors="pt",
                                                              add_special_tokens=False,
                                                              truncation=False)
                    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
                    convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
                    prompt_tokens = prompt_tokens.squeeze(0)

                    # Calculate where to inject the image
                    eot_id_indices = \
                        (convo_tokens == self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(
                            as_tuple=True)[
                            0].tolist()
                    assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

                    preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt

                    # Embed the tokens
                    convo_embeds = self.llm.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

                    # Embed image
                    # This results in Batch x Image Tokens x Features
                    with torch.amp.autocast_mode.autocast(device, enabled=True):
                        vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                        embedded_images = self.image_adapter(vision_outputs.hidden_states)
                        embedded_images = embedded_images.to(device)

                    # Construct the input
                    input_embeds = torch.cat([
                        convo_embeds[:, :preamble_len],  # Part before the prompt
                        embedded_images.to(dtype=convo_embeds.dtype),  # Image
                        convo_embeds[:, preamble_len:],  # The prompt and anything after it
                    ], dim=1).to(device)

                    input_ids = torch.cat([
                        convo_tokens[:preamble_len].unsqueeze(0),
                        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                        # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
                        convo_tokens[preamble_len:].unsqueeze(0),
                    ], dim=1).to(device)
                    attention_mask = torch.ones_like(input_ids)

                    # Debugging
                    self.logger.debug(f"Input to model: {repr(self.llm_tokenizer.decode(input_ids[0]))}")

                    # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
                    # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)
                    generate_ids = self.llm.generate(input_ids, inputs_embeds=input_embeds,
                                                     attention_mask=attention_mask,
                                                     temperature=temperature, max_new_tokens=max_new_tokens,
                                                     do_sample=True,
                                                     suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9

                    # Trim off the prompt
                    generate_ids = generate_ids[:, input_ids.shape[1]:]
                    if generate_ids[0][-1] == self.llm_tokenizer.eos_token_id or generate_ids[0][
                        -1] == self.llm_tokenizer.convert_tokens_to_ids(
                        "<|eot_id|>"):
                        generate_ids = generate_ids[:, :-1]

                    content = self.llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=False,
                                                              clean_up_tokenization_spaces=False)[0]
                content = content.strip()

            content_list = str(content).split(".")
            unique_content = list(dict.fromkeys(content_list))
            unique_content = '.'.join(unique_content)
            return unique_content

    def inference(self):
        llm_inference(self)

    def unload_model(self) -> bool:
        image_adapter_unloaded = llm_unloaded = clip_model_unloaded = False
        # Unload Image Adapter
        if hasattr(self, "image_adapter") and self.image_adapter is not None:
            self.logger.info(f'Unloading Image Adapter...')
            start = time.monotonic()
            del self.image_adapter
            self.logger.info(f'Image Adapter unloaded in {time.monotonic() - start:.1f}s.')
            image_adapter_unloaded = True
        # Unload LLM
        if hasattr(self, "llm") and self.llm is not None:
            self.logger.info(f'Unloading LLM...')
            start = time.monotonic()
            del self.llm
            if hasattr(self, "llm_tokenizer"):
                del self.llm_tokenizer
            self.logger.info(f'LLM unloaded in {time.monotonic() - start:.1f}s.')
            llm_unloaded = True
        # Unload CLIP
        if hasattr(self, "clip_model") and self.clip_model is not None:
            self.logger.info(f'Unloading CLIP...')
            start = time.monotonic()
            del self.clip_model
            del self.clip_processor
            self.logger.info(f'CLIP unloaded in {time.monotonic() - start:.1f}s.')
            clip_model_unloaded = True

        try:
            import torch
            if not self.args.llm_use_cpu:
                self.logger.debug(f'Will empty cuda device cache...')
                torch.cuda.empty_cache()
        except ImportError as ie:
            self.logger.error(f'Import torch Failed!\nDetails: {ie}')
            raise ImportError

        return image_adapter_unloaded and llm_unloaded and clip_model_unloaded
