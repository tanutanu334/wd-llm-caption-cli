import csv
import os
import time
from argparse import Namespace
from pathlib import Path

import numpy
from PIL import Image
from tqdm import tqdm

from . import get_caption_file_path, kaomojis
from ..utils.image_process_util import image_process, image_process_gbr, get_image_paths
from ..utils.logger_util import Logger


class Tagger:
    def __init__(
            self,
            logger: Logger,
            args: Namespace,
            model_path: Path,
            tags_csv_path: Path
    ):
        self.logger = logger
        self.args = args

        self.ort_infer_sess = None
        self.model_path = model_path
        self.tags_csv_path = tags_csv_path
        self.model_shape_size = None

        self.tag_freq = {}
        self.rating_tags = None
        self.character_tags = None
        self.general_tags = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.logger.error(f'{str(self.model_path)} NOT FOUND!')
            raise FileNotFoundError
        # Import ONNX
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as ie:
            self.logger.error(f'Import ONNX Failed!\nDetails: {ie}')
            raise ImportError

        self.logger.info(f'Loading model from {str(self.model_path)}')
        provider_options = None
        if 'CUDAExecutionProvider' in ort.get_available_providers() and not self.args.wd_force_use_cpu:
            providers = (['CUDAExecutionProvider'])
            self.logger.info('Use CUDA device for WD Tagger inference')
        elif 'ROCMExecutionProvider' in ort.get_available_providers() and not self.args.wd_force_use_cpu:
            providers = (['ROCMExecutionProvider'])
            self.logger.info('Use ROCM device for WD Tagger inference')
        elif "OpenVINOExecutionProvider" in ort.get_available_providers() and not self.args.wd_force_use_cpu:
            providers = (["OpenVINOExecutionProvider"])
            provider_options = [{'device_type': "GPU_FP32"}]
            self.logger.info('Use OpenVINO device for WD Tagger inference')
        else:
            if self.args.wd_force_use_cpu:
                self.logger.warning('wd_force_use_cpu ENABLED, will only use cpu for WD Tagger inference!')
            else:
                self.logger.info('Use CPU for WD Tagger inference')
                self.args.wd_force_use_cpu = True
            providers = (['CPUExecutionProvider'])

        self.logger.info(f'Loading {self.args.wd_model_name} with {"CPU" if self.args.wd_force_use_cpu else "GPU"}...')
        start_time = time.monotonic()
        self.ort_infer_sess = ort.InferenceSession(
            self.model_path,
            providers=providers,
            provider_options=provider_options
        )
        self.logger.info(f'{self.args.wd_model_name} Loaded in {time.monotonic() - start_time:.1f}s.')
        self.model_shape_size = self.ort_infer_sess.get_inputs()[0].shape[1]
        self.logger.debug(f'"{self.args.wd_model_name}" target shape is {self.model_shape_size}')

    def get_tags(
            self,
            image: Image.Image
    ) -> tuple[str, str, str, str]:
        tags_csv_path = self.tags_csv_path
        if not os.path.exists(tags_csv_path):
            self.logger.error(f'{str(tags_csv_path)} NOT FOUND!')
            raise FileNotFoundError

        self.logger.debug(f'Loading tags from {tags_csv_path}')
        with open(tags_csv_path, 'r', encoding='utf-8') as csv_file:
            csv_content = csv.reader(csv_file)
            rows = [row for row in csv_content]
            header = rows[0]
            rows = rows[1:]

        if not (header[0] in ("tag_id", "id") and header[1] == "name" and header[2] == "category"):
            self.logger.error(f'Unexpected csv header: {header}')
            raise ValueError

        if self.args.wd_model_name.lower().startswith("wd"):
            rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
            character_tags = [row[1] for row in rows[0:] if row[2] == "4"]
            general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
        else:
            self.logger.warning(f"{self.args.wd_model_name} doesn't support rating tags and character tags.")
            rating_tags = None
            character_tags = None
            general_tags = [row[1] for row in rows[0:]]

        if self.args.wd_character_tag_expand:
            if self.args.wd_model_name.lower().startswith("wd"):
                self.logger.info(
                    'character_tag_expand Enabled. character tags will be expanded like `character_name, series`.')

                for i, tag in enumerate(character_tags):
                    if tag.endswith(")"):
                        tags = tag.split("(")
                        character_tag = "(".join(tags[:-1])

                        if character_tag.endswith("_"):
                            character_tag = character_tag[:-1]
                        series_tag = tags[-1].replace(")", "")

                        character_tags[i] = character_tag + self.args.wd_caption_separator + series_tag
            else:
                self.logger.warning(f"{self.args.wd_model_name} doesn't support and character tags.")

        if self.args.wd_remove_underscore:
            self.logger.info('wd_remove_underscore Enabled. `_` will be replace to ` `.')
            if self.args.wd_model_name.lower().startswith("wd"):
                rating_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                               rating_tags]

                character_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                                  character_tags]

            general_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                            general_tags]

        if self.args.wd_tag_replacement:
            # escape , and ; in tag_replacement: wd14 tag names may contain , and ;
            escaped_tag_replacements = self.args.wd_tag_replacement.replace("\\,", "@@@@").replace("\\;", "####")
            tag_replacements = escaped_tag_replacements.split(";")

            for tag_replacement in tag_replacements:
                tags = tag_replacement.split(",")  # source, target

                if not len(tags) == 2:
                    self.logger.error(
                        f'tag replacement must be in the format of `source,target` : {self.args.wd_tag_replacement}')
                    raise ValueError

                source, target = [tag.replace("@@@@", ",").replace("####", ";") for tag in tags]
                self.logger.info(f'replacing tag: {source} -> {target}')

                if source in general_tags:
                    general_tags[general_tags.index(source)] = target
                elif source in character_tags and self.args.wd_model_name.lower().startswith("wd"):
                    character_tags[character_tags.index(source)] = target
                elif source in rating_tags and self.args.wd_model_name.lower().startswith("wd"):
                    rating_tags[rating_tags.index(source)] = target

        caption_separator = self.args.wd_caption_separator
        stripped_caption_separator = caption_separator.strip()
        undesired_tags = self.args.wd_undesired_tags.split(stripped_caption_separator)
        undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

        always_first_tags = [tag for tag in self.args.wd_always_first_tags.split(stripped_caption_separator)
                             if tag.strip() != ""] if self.args.wd_always_first_tags else None

        input_name = self.ort_infer_sess.get_inputs()[0].name
        label_name = self.ort_infer_sess.get_outputs()[0].name

        image = image_process(image, self.model_shape_size)
        self.logger.debug(f"Resized image shape: {image.shape}")
        image = image_process_gbr(image)
        image = numpy.array([image])
        prob = self.ort_infer_sess.run([label_name], {input_name: image})[0]  # onnx output numpy
        prob = prob[:len([image])][0]

        # def mcut_threshold(probs):
        #     """
        #     Maximum Cut Thresholding (MCut)
        #     Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
        #     for Multi-label Classification. In 11th International Symposium, IDA 2012
        #     (pp. 172-183).
        #     """
        #     sorted_probs = probs[probs.argsort()[::-1]]
        #     difs = sorted_probs[:-1] - sorted_probs[1:]
        #     t = difs.argmax()
        #     mcut_threshold = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        #     return mcut_threshold

        if not self.args.wd_model_name.lower().startswith("wd"):
            self.logger.warning(
                f'"{self.args.wd_model_name}" don\'t support general_threshold and character_threshold, '
                f'will set them to threshold value')
            self.args.wd_general_threshold = None
            self.args.wd_character_threshold = None

        self.logger.debug(f'threshold: {self.args.wd_threshold}') \
            if not self.args.wd_general_threshold and not self.args.wd_character_threshold else None
        self.logger.debug(f'General threshold: {self.args.wd_general_threshold}') \
            if self.args.wd_general_threshold else None
        self.logger.debug(f'Character threshold: {self.args.wd_character_threshold}') \
            if self.args.wd_character_threshold else None

        # Set general_threshold and character_threshold to general_threshold if not they are not set
        self.args.wd_general_threshold = self.args.wd_threshold if self.args.wd_general_threshold is None else self.args.wd_general_threshold
        self.args.wd_character_threshold = self.args.wd_threshold \
            if self.args.wd_character_threshold is None and self.args.wd_model_name.lower().startswith(
            "wd") else self.args.wd_character_threshold

        # if self.args.wd_maximum_cut_threshold:
        #     self.logger.debug('maximum_cut_threshold ENABLED!, all threshold will be overwritten.')
        #     general_prob = prob[len(rating_tags):(len(rating_tags)+len(general_tags))]
        #     general_prob = list(zip(general_tags, general_prob.astype(float)))
        #     general_prob = numpy.array([x[1] for x in general_prob])
        #
        #     character_prob = prob[len(rating_tags)+len(general_tags):]
        #     character_prob = list(zip(character_tags, character_prob.astype(float)))
        #     character_prob = numpy.array([x[1] for x in character_prob])
        #
        #     general_threshold = mcut_threshold(general_prob)
        #     self.logger.debug(f'general_threshold changed from '
        #                       f'{self.args.wd_general_threshold} to {general_threshold}')
        #     self.args.wd_general_threshold = general_threshold
        #
        #     character_threshold = max(0.15, mcut_threshold(character_prob))
        #     self.logger.debug(f'character_threshold changed from '
        #                       f'{self.args.wd_character_threshold} to {character_threshold}')
        #     self.args.wd_character_threshold = character_threshold

        combined_tags = []
        rating_tag_text = ""
        character_tag_text = ""
        general_tag_text = ""

        # First 4 labels are ratings, the rest are tags: pick anywhere prediction confidence >= threshold
        for i, p in enumerate(prob[len(rating_tags):] if self.args.wd_model_name.lower().startswith("wd") else prob):
            if i < len(general_tags) and p >= self.args.wd_general_threshold:
                tag_name = general_tags[i]

                if tag_name not in undesired_tags:
                    if self.args.wd_tags_frequency:
                        self.tag_freq[tag_name] = self.tag_freq.get(tag_name, 0) + 1

                    general_tag_text += caption_separator + tag_name
                    combined_tags.append(tag_name)

            elif (self.args.wd_character_threshold and i >= len(
                    general_tags) and p >= self.args.wd_character_threshold):
                tag_name = character_tags[i - len(general_tags)]

                if tag_name not in undesired_tags:
                    if self.args.wd_tags_frequency:
                        self.tag_freq[tag_name] = self.tag_freq.get(tag_name, 0) + 1

                    character_tag_text += caption_separator + tag_name

                    if self.args.wd_character_tags_first:  # insert to the beginning
                        combined_tags.insert(0, tag_name)

                    else:
                        combined_tags.append(tag_name)

        # First 4 labels are actually ratings: pick one with argmax
        if self.args.wd_add_rating_tags_to_first or self.args.wd_add_rating_tags_to_last:
            if self.args.wd_model_name.lower().startswith("wd"):
                ratings_probs = prob[:4]
                rating_index = ratings_probs.argmax()
                found_rating = rating_tags[rating_index]

                if found_rating not in undesired_tags:
                    if self.args.wd_tags_frequency:
                        self.tag_freq[found_rating] = self.tag_freq.get(found_rating, 0) + 1
                    rating_tag_text = found_rating
                    if self.args.wd_add_rating_tags_to_first:
                        combined_tags.insert(0, found_rating)  # insert to the beginning
                    else:
                        combined_tags.append(found_rating)
            else:
                self.logger.warning(f"{self.args.wd_model_name} doesn't support rating tags.")

        # Always put some tags at the beginning
        if always_first_tags is not None:
            for tag in always_first_tags:
                if tag in combined_tags:
                    combined_tags.remove(tag)
                    combined_tags.insert(0, tag)

        if len(general_tag_text) > 0:
            general_tag_text = general_tag_text[len(caption_separator):]

        if len(character_tag_text) > 0:
            character_tag_text = character_tag_text[len(caption_separator):]

        tag_text = caption_separator.join(combined_tags)

        return tag_text, rating_tag_text, character_tag_text, general_tag_text

    def inference(self):
        image_paths = get_image_paths(logger=self.logger, path=Path(self.args.data_path), recursive=self.args.recursive)
        pbar = tqdm(total=len(image_paths), smoothing=0.0)
        for image_path in image_paths:
            try:
                pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                             image_path[:15]) + ' ... ' + image_path[-20:])

                wd_caption_file = get_caption_file_path(
                    self.logger,
                    data_path=self.args.data_path,
                    image_path=Path(image_path),
                    custom_caption_save_path=self.args.custom_caption_save_path,
                    caption_extension=self.args.wd_caption_extension \
                        if self.args.caption_method == "wd+llm" else self.args.caption_extension
                )
                # Skip exists
                if self.args.skip_exists and os.path.isfile(wd_caption_file):
                    self.logger.warning(f'`skip_exists` ENABLED!!! '
                                        f'WD Caption file {wd_caption_file} already exists, Skip this caption.')
                    pbar.update(1)
                    continue
                # Image process
                image = Image.open(image_path)
                # Get tags
                tag_text, rating_tag_text, character_tag_text, general_tag_text = self.get_tags(
                    image=image
                )

                if not (self.args.not_overwrite and os.path.isfile(wd_caption_file)):
                    with open(wd_caption_file, "wt", encoding="utf-8") as f:
                        f.write(tag_text + "\n")

                    self.logger.debug(f"Image path: {image_path}")
                    self.logger.debug(f"Caption path: {wd_caption_file}")
                    if self.args.wd_model_name.lower().startswith("wd"):
                        self.logger.debug(f"Rating tags: {rating_tag_text}")
                        self.logger.debug(f"Character tags: {character_tag_text}")
                    self.logger.debug(f"General tags: {general_tag_text}")
                else:
                    self.logger.warning(f'`not_overwrite` ENABLED!!! '
                                        f'WD Caption file {wd_caption_file} already exist! Skip this caption.')

            except Exception as e:
                self.logger.error(f"Failed to caption image: {image_path}, skip it.\nerror info: {e}")
                pbar.update(1)
                continue

            pbar.update(1)
        pbar.close()

        if self.args.wd_tags_frequency:
            sorted_tags = sorted(self.tag_freq.items(), key=lambda x: x[1], reverse=True)
            self.logger.info('Tag frequencies:')
            for tag, freq in sorted_tags:
                self.logger.info(f'{tag}: {freq}')

    def unload_model(self) -> bool:
        unloaded = False
        if self.ort_infer_sess:
            self.logger.info(f'Unloading model {self.args.wd_model_name}...')
            start = time.monotonic()
            del self.ort_infer_sess
            if self.rating_tags:
                del self.rating_tags
            if self.character_tags:
                del self.character_tags
            if self.general_tags:
                del self.general_tags
            self.logger.info(f'{self.args.wd_model_name} unloaded in {time.monotonic() - start:.1f}s.')
            del self.model_path
            del self.tags_csv_path
            del self.args

            unloaded = True

        return unloaded
