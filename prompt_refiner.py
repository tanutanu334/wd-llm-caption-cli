import argparse
import json
from pathlib import Path

from diffusers import StableDiffusionPipeline
from PIL import Image

from wd_llm_caption.caption import Caption, setup_args


def generate_image(prompt: str, model: str, output: Path) -> Path:
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype="auto")
    pipe = pipe.to("cuda" if pipe.device.type == "cuda" else "cpu")
    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save(output)
    return output


def caption_image(image_path: Path) -> str:
    args = setup_args()
    args.data_path = str(image_path)
    args.caption_method = "llm"
    args.llm_choice = "joy"
    args.llm_model_name = "Joy-Caption-Beta-One-Llava"
    args.llm_user_prompt = "Please describe this image."
    args.llm_system_prompt = "You are a caption expert." \
        "Describe all visual details."  # short system prompt
    args.run_method = "sync"
    args.recursive = False
    cap = Caption()
    cap.check_path(args)
    cap.set_logger(args)
    cap.download_models(args)
    cap.load_models(args)
    cap.my_llm.args = args
    image = Image.open(image_path)
    caption = cap.my_llm.get_caption(
        image=image,
        system_prompt=args.llm_system_prompt,
        user_prompt=args.llm_user_prompt,
    )
    cap.unload_models()
    return caption


def refine_prompt(original: str, caption: str) -> str:
    orig_set = set(original.lower().split())
    caption_set = set(caption.lower().split())
    missing = caption_set - orig_set
    improved = original + ", " + ", ".join(sorted(missing)) if missing else original
    return improved


def main() -> None:
    parser = argparse.ArgumentParser(description="Improve prompt by diffing caption")
    parser.add_argument("prompt", help="initial prompt")
    parser.add_argument("--sd_model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--output", default="generated.png")
    args = parser.parse_args()

    output = Path(args.output)
    generate_image(args.prompt, args.sd_model, output)
    caption = caption_image(output)
    improved = refine_prompt(args.prompt, caption)
    result = {
        "original": args.prompt,
        "caption": caption,
        "improved": improved,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
