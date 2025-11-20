#!/usr/bin/env python3
"""
Generate 'camp' and 'school' images using the ORIGINAL SD-1.5 model only.
Outputs: camp.png, school.png in current directory.
"""

import torch
from diffusers import StableDiffusionPipeline

def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    prompts = ["camp", "school"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading Stable Diffusion 1.5 from: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(42)

    for prompt in prompts:
        print(f"Generating: {prompt!r}")

        out = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator
        )
        img = out.images[0]

        filename = f"{prompt}.png"
        img.save(filename)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
