import os
import torch
import datetime
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from PIL import Image

from tqdm.auto import tqdm


preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def transform_fn(examples):
    processed_images = []
    for image in examples["image"]:
        if isinstance(image, str):
            with Image.open(image) as img:
                img = img.convert("RGB")
                processed_images.append(preprocess(img))
        elif isinstance(image, Image.Image):
            processed_images.append(preprocess(image.convert("RGB")))
        else:
            raise ValueError("Unknown image type in dataset: {}".format(type(image)))
    return {"pixel_values": processed_images}


def train_arcane_style(config):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = os.path.join(config["logs_dir"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    dataset = load_dataset(
        "imagefolder",
        data_dir=config["data_dir"],
        split="train"
    )
    dataset.set_transform(transform_fn)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2
    )
    
    weight_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        config["model_id"],
        torch_dtype=weight_dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    
    lora_config = LoraConfig(
        inference_mode=False,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    
    lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=config["learning_rate"])
    
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    pipe.unet.train()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    
    global_step = 0
    for epoch in range(config["num_epochs"]):
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            images = batch["pixel_values"].to(device, dtype=weight_dtype)
            global_step += 1
            
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
            
            text_input = pipe.tokenizer(
                [config["prompt"]] * config["batch_size"],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (config["batch_size"],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)
            
            epoch_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        progress_bar.close()
        
        if (epoch) % 10 == 0:
            checkpoint_dir = f"checkpoint-arcane-{epoch+1}"
            pipe.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint_dir}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train arcane style model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--data_dir", type=str, default="./data/images", help="Directory with training images")
    parser.add_argument("--logs_dir", type=str, default="./logs/arcane_finetune", help="Directory with train logs")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Model id to be used")
    parser.add_argument("--prompt", type=str, default="arcane style, detailed, high quality, League of Legends", help="Prompt text")
    args = parser.parse_args()
    
    config = vars(args)
    train_arcane_style(config)
    