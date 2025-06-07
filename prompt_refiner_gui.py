import gradio as gr
from pathlib import Path
from prompt_refiner import generate_image, caption_image, refine_prompt


def run(prompt: str, sd_model: str):
    output = Path("generated.png")
    generate_image(prompt, sd_model, output)
    caption = caption_image(output)
    improved = refine_prompt(prompt, caption)
    return output, caption, improved


def gui() -> None:
    with gr.Blocks(title="Prompt Refinement") as demo:
        gr.Markdown("## Prompt Refinement")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=1)
            sd_model = gr.Textbox(label="Stable Diffusion Model", value="stabilityai/stable-diffusion-2-1-base")
        run_btn = gr.Button("Generate and Refine")
        image_out = gr.Image(label="Generated Image")
        caption_out = gr.Text(label="Caption", show_copy_button=True)
        improved_out = gr.Text(label="Improved Prompt", show_copy_button=True)
        run_btn.click(fn=run, inputs=[prompt, sd_model], outputs=[image_out, caption_out, improved_out])
    demo.launch()


if __name__ == "__main__":
    gui()
