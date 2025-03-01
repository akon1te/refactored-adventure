import gradio as gr
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline


MODEL_PATH = "fine-tuned-sd-controlnet"  # Измените на нужный путь, если требуется
device = "msp" if torch.mps.is_available() else "cpu"

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_PATH,
    safety_checker=None,
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)
pipe.to(device)

def generate_image(prompt, init_image, strength, num_inference_steps):
    # Убедимся, что изображение имеет нужный размер
    init_image = init_image.convert("RGB").resize((512, 512))
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps
    )
    return result.images[0]

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.inputs.Textbox(label="Prompt", default="arcane style"),
        gr.inputs.Image(type="pil", label="Исходное изображение"),
        gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.05, default=0.75, label="Strength"),
        gr.inputs.Slider(minimum=10, maximum=100, step=5, default=50, label="Inference Steps"),
    ],
    outputs=gr.outputs.Image(type="pil", label="Сгенерированное изображение"),
    title="Stable Diffusion ControlNet Demo",
    description="Демо для генерации изображений с дообученной моделью Stable Diffusion с ControlNet"
)

if __name__ == "__main__":
    demo.launch()