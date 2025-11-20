"""
Generate pictograms using Stable Diffusion img2img with optional LoRA weights.
Takes an input image and transforms it according to the text prompt.
"""
import argparse
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from PIL import Image
import os


def load_pipeline(base_model_path, lora_weights_path=None, device="cuda"):
    """
    Load Stable Diffusion img2img pipeline with optional LoRA weights.

    Args:
        base_model_path: Path to base SD model
        lora_weights_path: Path to trained LoRA weights (optional)
        device: Device to load model on
    """
    print(f"Loading base model from {base_model_path}...")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable for faster inference
    )

    # Use faster scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    if lora_weights_path is not None:
        print(f"Loading LoRA weights from {lora_weights_path}...")
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            lora_weights_path
        )
    else:
        print("Using base model without LoRA weights")

    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()  # Memory optimization

    return pipeline


def generate_pictogram_img2img(
    pipeline,
    input_image,
    prompt,
    negative_prompt="blurry, photo, photograph, realistic, complex, detailed background",
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=None,
    instance_token=None
):
    """
    Generate a pictogram from an input image and text prompt.

    Args:
        pipeline: Loaded StableDiffusionImg2ImgPipeline with LoRA
        input_image: PIL Image to transform
        prompt: Text description of desired pictogram style/content
        negative_prompt: What to avoid in generation
        strength: How much to transform the input (0.0-1.0, higher = more change)
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility
        instance_token: Optional instance token to prepend to prompt (e.g., "sks")
    """
    # Prepend instance token if provided
    if instance_token is not None:
        prompt = f"{instance_token} {prompt}"

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    with torch.inference_mode():
        image = pipeline(
            prompt=prompt,
            image=input_image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Generate pictograms with Stable Diffusion img2img (with optional LoRA)"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to trained LoRA weights (optional - use base model if not provided)"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image to transform"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for pictogram generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_pictogram_img2img.png",
        help="Output filename"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Transformation strength (0.0-1.0). Lower = closer to input, higher = more creative"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output image height (input will be resized to this, should match training resolution)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Output image width (input will be resized to this, should match training resolution)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--instance_token",
        type=str,
        default=None,
        help="Instance token to prepend to prompt (e.g., 'sks' for LoRA fine-tuned models)"
    )

    args = parser.parse_args()

    # Validate strength parameter
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("Strength must be between 0.0 and 1.0")

    # Load input image
    print(f"Loading input image from {args.input_image}...")
    input_image = Image.open(args.input_image).convert("RGB")
    original_size = input_image.size
    print(f"Original image size: {original_size}")

    # Resize to target dimensions to avoid OOM
    target_size = (args.width, args.height)
    if original_size != target_size:
        print(f"Resizing to: {target_size}")
        input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)

    # Load pipeline
    pipeline = load_pipeline(
        args.base_model,
        args.lora_weights,
        args.device
    )

    # Prepare final prompt with instance token if provided
    final_prompt = f"{args.instance_token} {args.prompt}" if args.instance_token else args.prompt

    print(f"\nGenerating {args.num_images} pictogram(s) for: '{final_prompt}'")
    print(f"Negative prompt: '{args.negative_prompt}'")
    print(f"Strength: {args.strength}")
    if args.instance_token:
        print(f"Using instance token: '{args.instance_token}'")

    # Generate images
    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None

        print(f"\nGenerating image {i+1}/{args.num_images}...")
        image = generate_pictogram_img2img(
            pipeline,
            input_image,
            args.prompt,
            args.negative_prompt,
            args.strength,
            args.steps,
            args.guidance_scale,
            seed,
            args.instance_token
        )

        # Save image
        if args.num_images == 1:
            output_path = args.output
        else:
            name, ext = os.path.splitext(args.output)
            output_path = f"{name}_{i+1}{ext}"

        image.save(output_path)
        print(f"Saved to: {output_path}")

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
