import argparse
import os
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline

def main(args):
    device = "mps" if torch.mps.is_available() else "cpu"

    # Загружаем дообученную модель
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_path,
        safety_checker=None,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    )
    pipe.to(device)

    # Загружаем исходное изображение для инференса
    init_image = Image.open(args.init_image).convert("RGB")
    init_image = init_image.resize((512, 512))

    # Генерируем изображение на основе текста и исходного изображения
    result = pipe(
        prompt=args.prompt,
        image=init_image,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps
    )
    out_image = result.images[0]
    out_image.save(args.out_image)
    print(f"Сгенерированное изображение сохранено: {args.out_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference для дообученной модели Stable Diffusion с ControlNet"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Путь к дообученной модели")
    parser.add_argument("--init_image", type=str, required=True, help="Путь к исходному изображению")
    parser.add_argument("--prompt", type=str, default="arcane style", help="Текстовое описание для генерации")
    parser.add_argument("--strength", type=float, default=0.75, help="Степень преобразования исходного изображения")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Количество шагов инференса")
    parser.add_argument("--out_image", type=str, default="generated.png", help="Путь для сохранения сгенерированного изображения")
    
    args = parser.parse_args()
    main(args)