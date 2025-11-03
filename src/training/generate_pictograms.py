"""
Generate pictograms using Stable Diffusion with optional LoRA weights.
"""
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import os


def load_pipeline(base_model_path, lora_weights_path=None, device="cuda"):
    """
    Load Stable Diffusion pipeline with optional LoRA weights.

    Args:
        base_model_path: Path to base SD model
        lora_weights_path: Path to trained LoRA weights (optional)
        device: Device to load model on
    """
    print(f"Loading base model from {base_model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
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


def generate_pictogram(
    pipeline,
    prompt,
    negative_prompt="blurry, photo, photograph, realistic, complex, detailed background",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=256,
    width=256,
    seed=None,
    instance_token=None
):
    """
    Generate a pictogram from a text prompt.

    Args:
        pipeline: Loaded StableDiffusionPipeline with LoRA
        prompt: Text description of pictogram to generate
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        height: Image height in pixels
        width: Image width in pixels
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
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate pictograms with Stable Diffusion (with optional LoRA)")

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
        default="generated_pictogram.png",
        help="Output filename"
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
        help="Image height (should match training resolution)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Image width (should match training resolution)"
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
    if args.instance_token:
        print(f"Using instance token: '{args.instance_token}'")

    # Generate images
    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None

        print(f"\nGenerating image {i+1}/{args.num_images}...")
        image = generate_pictogram(
            pipeline,
            args.prompt,
            args.negative_prompt,
            args.steps,
            args.guidance_scale,
            args.height,
            args.width,
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
