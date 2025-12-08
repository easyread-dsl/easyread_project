# ARASAAC Pictogram Generation with LoRA

This pipeline trains a LoRA adapter on Stable Diffusion 1.5 to generate ARASAAC-style pictograms for new concepts.

## Overview

The pipeline consists of:
1. Data preparation - Convert ARASAAC dataset into training format
2. LoRA training - Finetune SD 1.5 with LoRA adapters
3. Inference - Generate new pictograms using trained model

## Dependencies

Install the following packages:

```bash
pip install torch torchvision torchaudio
pip install diffusers==0.25.0
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install datasets
pip install Pillow
pip install tqdm
```

For GPU training, ensure you have CUDA installed.

### Hardware Requirements

- GPU: Minimum 12GB VRAM (16GB+ recommended)
- RAM: 16GB+ system RAM
- Storage: ~5GB for model weights, ~1GB for dataset

## Usage

### Step 1: Prepare Dataset

First, run the data preparation script to convert ARASAAC data into training format:

```bash
python prepare_dataset.py
```

This will:
- Load images from the ARASAAC dataset
- Generate captions (you can modify caption style in the script)
- Save processed data to `./training_data/`

**Caption styles available:**
- `simple`: Just the title (e.g., "grandfather")
- `descriptive`: Natural description (e.g., "a pictogram of grandfather, elderly family member")
- `template`: Consistent format (e.g., "ARASAAC pictogram showing grandfather")

Edit `prepare_dataset.py` line ~145 to change caption style:
```python
prepare_training_data(
    caption_style="descriptive",  # Change to "simple" or "template"
    max_samples=None  # Set to small number for quick testing
)
```

### Step 2: Train LoRA Model

Run training with the provided configuration:

```bash
bash train_config.sh
```

Or customize training parameters:

```bash
python train_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --data_dir="./training_data" \
    --output_dir="./lora_output" \
    --resolution=512 \
    --train_batch_size=4 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --lora_rank=4 \
    --lora_alpha=4 \
    --mixed_precision="fp16" \
    --save_steps=500
```

**Key parameters to adjust:**

- `--train_batch_size`: Reduce to 2 or 1 if out of memory
- `--gradient_accumulation_steps`: Increase to compensate for smaller batch size
- `--num_train_epochs`: More epochs for better learning (100-200 typical)
- `--lora_rank`: Higher rank (8, 16) = more capacity but slower (4 is good start)
- `--learning_rate`: 1e-4 is standard, try 5e-5 for more stable training

Training will save checkpoints to `./lora_output/checkpoint-{step}/` and final model to `./lora_output/final/`.

**Expected training time:**
- ~3876 samples with batch size 4: ~970 steps per epoch
- 100 epochs = ~97,000 steps
- On A100: ~2-3 hours
- On RTX 3090: ~4-6 hours

### Step 3: Generate Pictograms

Generate new pictograms using trained model:

```bash
python generate_pictograms.py \
    --lora_weights="./lora_output/final" \
    --prompt="a pictogram of a teacher" \
    --output="teacher.png" \
    --num_images=4 \
    --seed=42
```


```bash
python generate_pictograms.py \
    --lora_weights="/mnt/data2/easyread_project/results/lora_output_diverse_prompts_run4/checkpoint-37000" \
    --prompt="a person on a rock with a blue shirt and a red hat; background color: yellow; skin color: black; hair color: blonde" \
    --output="/mnt/data2/outputs/persononrock_general.png" \
    --num_images=4 \
    --seed=42 \
    --instance_token="sks"
```

SKIN_COLORS = ["white", "black", "assian", "mulatto", "aztec"]
HAIR_COLORS = ["blonde", "brown", "darkBrown", "gray", "darkGray", "red", "black"]
BACKGROUND_COLORS = {
    "red": "FF0000",
    "green": "00FF00",
    "blue": "0000FF",
    "yellow": "FFFF00",
    "black": "000000",
    "white": "FFFFFF",
}

**Parameters:**
- `--lora_weights`: Path to trained LoRA weights (use checkpoint or final)
- `--prompt`: Description of pictogram to generate
- `--negative_prompt`: What to avoid (defaults work well for pictograms)
- `--num_images`: Generate multiple variations
- `--steps`: More steps = higher quality (30 is good, 50 for best)
- `--guidance_scale`: How closely to follow prompt (7.5 default, try 5-10)
- `--seed`: For reproducible results

**Example prompts:**
```bash
# Simple concept
--prompt="a pictogram of a teacher"

# With attributes
--prompt="a pictogram of a happy teacher"

# Composite concepts
--prompt="a pictogram of a teacher reading a book"

# Use similar style to training data
--prompt="ARASAAC pictogram showing a nurse"
```

## Tips for Best Results

### Training Tips

1. **Monitor loss**: Loss should decrease steadily. If it plateaus too early, try:
   - Increasing learning rate slightly
   - Training for more epochs
   - Increasing LoRA rank

2. **Prevent overfitting**: If generated images look too similar to training data:
   - Reduce number of epochs
   - Lower learning rate
   - Add more diverse training captions

3. **Caption quality matters**: The better your captions describe the pictograms, the better the model learns. Consider spending time on caption engineering.

4. **Checkpointing**: Test different checkpoints - sometimes earlier checkpoints generalize better than the final model.

### Generation Tips

1. **Prompt engineering**:
   - Start prompts similar to training captions
   - Use "a pictogram of..." or "ARASAAC pictogram showing..."
   - Keep prompts simple and clear

2. **Negative prompts**: Use these to maintain pictogram style:
   - "blurry, photo, photograph, realistic, complex, detailed background"
   - Add "colorful, multiple colors" if you want pure black/white

3. **Guidance scale**:
   - Lower (5-7): More creative, might deviate from style
   - Higher (8-12): Stricter adherence to prompt and style

4. **Generate batches**: Create multiple images and pick the best ones

## File Structure

```
pipeline/
├── load_data.py              # Original data loading script
├── prepare_dataset.py        # Data preparation for training
├── train_lora.py            # LoRA training script
├── train_config.sh          # Training configuration
├── generate_pictograms.py   # Inference script
├── README.md                # This file
├── training_data/           # Prepared training data (created by prepare_dataset.py)
│   ├── images/
│   └── metadata.jsonl
└── lora_output/             # Training outputs (created by train_lora.py)
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── final/
```

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size: `--train_batch_size=2` or `--train_batch_size=1`
2. Increase gradient accumulation: `--gradient_accumulation_steps=2`
3. Use mixed precision: `--mixed_precision="fp16"`
4. Reduce resolution: `--resolution=256` (though 512 is recommended)

### Poor Generation Quality

1. Train for more epochs (100-200)
2. Check training loss - should be below 0.1 for good results
3. Try different checkpoints
4. Improve caption quality in preparation step
5. Increase LoRA rank: `--lora_rank=8`

### Training Too Slow

1. Increase batch size if you have memory: `--train_batch_size=8`
2. Use fewer epochs but higher learning rate
3. Reduce logging frequency: `--logging_steps=100`
4. Reduce save frequency: `--save_steps=1000`

## Advanced: Hyperparameter Tuning

If default settings don't work well, try these configurations:

**For faster experimentation (lower quality):**
```bash
--train_batch_size=8 \
--num_train_epochs=50 \
--learning_rate=2e-4 \
--lora_rank=4
```

**For best quality (slower):**
```bash
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--num_train_epochs=200 \
--learning_rate=5e-5 \
--lora_rank=8 \
--lora_alpha=8
```

**For low memory (~10GB):**
```bash
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--resolution=512 \
--mixed_precision="fp16"
```

## Next Steps

After training:

1. Test on various prompts to evaluate quality
2. Compare different checkpoints
3. Fine-tune hyperparameters based on results
4. Consider training longer if underfitting
5. Try different caption styles if results aren't good

## References

- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- LoRA: https://arxiv.org/abs/2106.09685
- Diffusers: https://github.com/huggingface/diffusers
- PEFT: https://github.com/huggingface/peft
