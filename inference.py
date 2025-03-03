import argparse
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load fine-tuned model
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_path,
        safety_checker=None,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
    )
    pipe = pipe.to(device)
    
    # Open and resize initial image
    init_image = Image.open(args.init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate image from initial image with given prompt and strength
    result = pipe(
        prompt=args.prompt,
        image=init_image,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps
    )
    
    # Save the generated image
    output_image = result.images[0]
    output_image.save(args.output)
    print(f"Generated image saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Stable Diffusion Img2Img model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--init_image", type=str, required=True, help="Path to the input image (PNG/JPG)")
    parser.add_argument("--prompt", type=str, default="arcane style, detailed, high quality, League of Legends", help="Text prompt")
    parser.add_argument("--strength", type=float, default=0.75, help="How much to transform the input image (0-1)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    
    args = parser.parse_args()
    main(args)