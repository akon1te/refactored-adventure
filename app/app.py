import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Load the fine-tuned model once at startup
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "path/to/your/fine-tuned-model"  # update this path

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_PATH,
    safety_checker=None,
    torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
)
pipe = pipe.to(device)

def generate_image(prompt, init_image, strength, num_inference_steps):
    # Ensure the initial image is in RGB and resized properly
    init_image = init_image.convert("RGB")
    init_image = init_image.resize((512, 512))
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=int(num_inference_steps)
    )
    return result.images[0]

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.inputs.Textbox(label="Prompt", default="arcane style, detailed, high quality, League of Legends"),
        gr.inputs.Image(type="pil", label="Input Image"),
        gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.75, label="Strength"),
        gr.inputs.Slider(minimum=10, maximum=100, step=1, default=50, label="Inference Steps")
    ],
    outputs=gr.outputs.Image(type="pil", label="Generated Image"),
    title="Stable Diffusion Img2Img Demo",
    description="Generate arcane style images using a fine-tuned Stable Diffusion model."
)

if __name__ == "__main__":
    demo.launch()