"""
LoRA training script for ARASAAC pictogram generation using Stable Diffusion 1.5
"""
import argparse
import os
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb


class FullDataset(Dataset):
    """Dataset for ARASAAC pictograms with captions."""

    def __init__(self, data_dir, tokenizer, size=512, instance_token="sks"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.size = size
        self.tokenizer = tokenizer
        self.instance_token = instance_token

        # Load metadata: support JSONL and JSON
        jsonl_path = self.data_dir / "metadata.jsonl"
        json_path = self.data_dir / "metadata.json"

        raw = []
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))
        elif json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, list):
                    raise ValueError("metadata.json must be a list of objects")
                raw = loaded
        else:
            raise FileNotFoundError("No metadata.jsonl or metadata.json found in data_dir")

        # Normalize fields so downstream code always has: file_name, text
        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                return [t.strip() for t in x.split("|") if t.strip()]
            return []

        self.data = []
        for row in raw:
            # file name: accept several variants
            fn = row.get("file_name") or row.get("image_file") or row.get("image") or row.get("path")
            if not fn:
                # skip entries without a usable filename
                continue
            # keep only the basename if a path was provided
            fn = os.path.basename(fn)

            # caption text
            text = row.get("text") or row.get("caption")
            if not text:
                # synthesize from title/keywords if missing
                title = row.get("title") or ""
                kws = _as_list(row.get("keywords"))
                kws = [k for k in kws if k and k != title][:7]
                if title and kws:
                    text = f"{title}, {', '.join(kws)}"
                elif title:
                    text = title
                elif kws:
                    text = ', '.join(kws)
                else:
                    raise ValueError(f"Cannot create caption for image {fn}: no text, title, or keywords found")

            # Prepend instance token to the caption
            text = f"{self.instance_token} {text}"

            # store normalized copy (preserve original fields too)
            nr = dict(row)
            nr["file_name"] = fn
            nr["text"] = text
            self.data.append(nr)

        if not self.data:
            raise ValueError("No valid samples found after normalizing metadata.")

        print(f"Loaded {len(self.data)} training samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import numpy as np  # safe local import
        item = self.data[idx]

        # Load and process image
        image_path = self.images_dir / item["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Resize and normalize to [-1, 1]
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize((self.size, self.size), resample=resample)

        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0  # HWC in [-1,1]
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Tokenize caption
        caption = item["text"]
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": image,
            "input_ids": input_ids,
        }





def collate_fn(examples):
    """Collate function for dataloader."""
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    input_ids = torch.stack([example['input_ids'] for example in examples])

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids
    }


def train(args):
    """Main training function."""

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize W&B
    if accelerator.is_main_process and not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=["lora", "stable-diffusion", "arasaac"]
        )
        print(f"W&B run initialized: {wandb.run.name}")

    # Load models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA layers to UNet
    print("Adding LoRA layers...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Log trainable parameters to W&B
    if accelerator.is_main_process and not args.no_wandb:
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in unet.parameters())
        wandb.config.update({
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": 100 * trainable_params / total_params
        })

    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = FullDataset(
        args.data_dir,
        tokenizer,
        size=args.resolution,
        instance_token=args.instance_token
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_train_steps,
        eta_min=args.learning_rate * 0.1
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move models to device
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device)

    # Resume from checkpoint if specified
    global_step = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

        # Load LoRA weights
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {args.resume_from_checkpoint}")

        # Load the LoRA adapter weights
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.load_adapter(args.resume_from_checkpoint, adapter_name="default")
        print(f"Loaded LoRA weights from {args.resume_from_checkpoint}")

        # Load training state (optimizer, scheduler, global_step)
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=accelerator.device)
            global_step = training_state['global_step']
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])
            print(f"Resumed from global step {global_step}")

            # Calculate starting epoch based on global_step
            num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / args.gradient_accumulation_steps
            )
            starting_epoch = global_step // num_update_steps_per_epoch
            print(f"Starting from epoch {starting_epoch}")
        else:
            print(f"Warning: training_state.pt not found in {args.resume_from_checkpoint}")
            print("Starting from step 0, but using loaded LoRA weights")

    # Training loop
    print("Starting training...")
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(starting_epoch, args.num_train_epochs):
        unet.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch['pixel_values'].to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                else:
                    grad_norm = None

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.detach().item()
                epoch_steps += 1

                # Log metrics
                if global_step % args.logging_steps == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logs = {
                        "train/loss": loss.detach().item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }

                    if grad_norm is not None:
                        logs["train/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                    progress_bar.set_postfix(**{"loss": logs["train/loss"], "lr": logs["train/learning_rate"]})

                    # Log to W&B
                    if accelerator.is_main_process and not args.no_wandb:
                        wandb.log(logs, step=global_step)

                # Save checkpoint and run validation
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_lora_weights(accelerator, unet, save_path, optimizer, lr_scheduler, global_step)

                    # Generate validation images
                    if accelerator.is_main_process:
                        print(f"\n{'='*50}")
                        print(f"Running validation at step {global_step}")
                        print(f"{'='*50}")

                        validation_images = generate_validation_images(
                            unet=unet,
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            noise_scheduler=noise_scheduler,
                            accelerator=accelerator,
                            instance_token=args.instance_token,
                            validation_prompts=args.validation_prompts,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            seed=args.seed
                        )

                        # Save validation images locally
                        val_dir = os.path.join(args.output_dir, f"validation-{global_step}")
                        os.makedirs(val_dir, exist_ok=True)

                        for prompt, image in validation_images:
                            # Clean prompt for filename (remove instance token and special chars)
                            clean_prompt = prompt.replace(args.instance_token, "").strip()
                            clean_prompt = "".join(c if c.isalnum() else "_" for c in clean_prompt)
                            image_path = os.path.join(val_dir, f"{clean_prompt}.png")
                            image.save(image_path)
                            print(f"Saved validation image: {image_path}")

                        # Log to W&B
                        if not args.no_wandb:
                            wandb_images = [
                                wandb.Image(image, caption=prompt)
                                for prompt, image in validation_images
                            ]
                            wandb.log({
                                "validation/images": wandb_images,
                                "train/checkpoint_saved": global_step
                            }, step=global_step)

                        print(f"{'='*50}\n")

            if global_step >= args.max_train_steps:
                break

        # Log epoch metrics
        if epoch_steps > 0 and accelerator.is_main_process and not args.no_wandb:
            avg_epoch_loss = epoch_loss / epoch_steps
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch_num": epoch
            }, step=global_step)

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        save_lora_weights(accelerator, unet, save_path)

        # Finish W&B run
        if not args.no_wandb:
            wandb.finish()

    print("Training complete!")


def save_lora_weights(accelerator, model, save_path, optimizer=None, lr_scheduler=None, global_step=None):
    """Save LoRA weights and training state."""
    os.makedirs(save_path, exist_ok=True)

    # Save LoRA weights
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path)

    # Save optimizer and scheduler state for resuming
    if optimizer is not None and lr_scheduler is not None and global_step is not None:
        training_state = {
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        state_path = os.path.join(save_path, "training_state.pt")
        torch.save(training_state, state_path)
        print(f"Saved training state to {state_path}")

    print(f"Saved LoRA weights to {save_path}")


def generate_validation_images(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    accelerator,
    instance_token,
    validation_prompts=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
):
    """Generate validation images for monitoring training progress."""
    if validation_prompts is None:
        validation_prompts = ["school", "train", "camp", "mother", "basketball"]

    # Prepend instance token to all prompts
    validation_prompts = [f"{instance_token} {prompt}" for prompt in validation_prompts]

    print(f"Generating validation images for prompts: {validation_prompts}")

    # Set models to eval mode
    unet.eval()
    text_encoder.eval()
    vae.eval()

    # Store generated images
    generated_images = []

    # Set seed for reproducibility
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    with torch.no_grad():
        for prompt in validation_prompts:
            # Tokenize prompt
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]

            # Prepare unconditional embeddings for classifier-free guidance
            uncond_input = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]

            # Concatenate for classifier-free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Prepare latents
            latents = torch.randn(
                (1, unet.config.in_channels, 64, 64),
                generator=generator,
                device=accelerator.device,
                dtype=text_embeddings.dtype
            )

            # Set timesteps
            noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
            latents = latents * noise_scheduler.init_noise_sigma

            # Denoising loop
            for t in noise_scheduler.timesteps:
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                # Predict noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

                # Perform classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous latent
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            # Decode latents to image
            latents = latents / vae.config.scaling_factor
            image = vae.decode(latents).sample

            # Convert to PIL Image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            image = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(image)

            generated_images.append((prompt, pil_image))

    # Set models back to train mode
    unet.train()

    return generated_images


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on ARASAAC pictograms")

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained model to use"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with prepared training data"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution"
    )
    parser.add_argument(
        "--instance_token",
        type=str,
        default="sks",
        help="Rare token to trigger the learned style (default: sks)"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha"
    )

    # Training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_train_epochs)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm"
    )

    # Optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="Output directory"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )

    # Validation arguments
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=["school", "train", "camp", "mother"],
        help="Prompts to use for validation image generation"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for validation image generation"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for validation image generation"
    )

    # W&B arguments
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="arasaac-lora",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or username)"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not provided)"
    )

    # Checkpoint resuming
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )

    return parser.parse_args()


if __name__ == "__main__":
    import numpy as np  # Import here for the dataset class

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
