import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler
)
from accelerate import Accelerator


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Device: {device}")
    
    # Параметры обучения из аргументов командной строки
    image_dir = args.image_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    prompt = args.prompt

    # Преобразования для изображений: изменение размера, нормализация и перевод в тензор.
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Загружаем модель ControlNet и Stable Diffusion pipeline
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32
    )
    pipe.to(device)

    # Переводим необходимые компоненты в режим обучения
    pipe.unet.train()
    pipe.controlnet.train()

    # Оптимизируем параметры UNet (при необходимости можно добавить и параметры controlnet)
    optimizer = optim.AdamW(pipe.unet.parameters(), lr=learning_rate)

    # Создаем scheduler для добавления шума
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Токенизатор и текстовый энкодер для получения эмбеддингов подсказки
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Подготавливаем модель и даталоадер с помощью Accelerator.
    # Обратите внимание, что мы подготавливаем только unet, optimizer и dataloader,
    # так как остальные компоненты (controlnet, vae, text_encoder) не обновляются.
    prepared_unet, optimizer, dataloader = accelerator.prepare(pipe.unet, optimizer, dataloader)
    pipe.unet = prepared_unet

    for epoch in range(num_epochs):
        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for images in progress_bar:
            images = images.to(device)

            # Кодирование текстового промпта в эмбеддинги
            text_inputs = tokenizer(
                [prompt] * images.shape[0],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

            # Кодирование изображений в латентное пространство через VAE
            latents = pipe.vae.encode(images).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            # Выбираем случайные временные шаги для каждого примера в батче
            batch_size_current = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size_current,), device=device)

            # Генерируем шум и добавляем его к латентам
            noise = torch.randn(latents.shape, device=device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Для условного управления ControlNet используем исходные изображения
            controlnet_conditioning = images

            # Предсказываем остаточный шум с помощью UNet, передавая эмбеддинги и условие от ControlNet
            model_output = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_conditioning=controlnet_conditioning
            )
            noise_pred = model_output.sample

            # Вычисляем MSE потерю между предсказанным и исходным шумом
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            progress_bar.set_description(f"Loss: {loss.item():.4f}")

    # Сохраняем дообученную модель
    accelerator.print(f"Сохранение модели в {args.save_path}...")
    pipe.save_pretrained(args.save_path)
    accelerator.print(f"Модель сохранена в папке {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning Stable Diffusion v1-5 с ControlNet для задачи image-to-image генерации с использованием Accelerate"
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Путь к директории с изображениями")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--num_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Параметр learning rate")
    parser.add_argument("--prompt", type=str, default="arcane style", help="Текстовое описание для обучения")
    parser.add_argument("--save_path", type=str, default="fine-tuned-sd-controlnet", help="Путь для сохранения дообученной модели")
    
    args = parser.parse_args()
    main(args)