# Finetuning Diffusion Models for EasyRead-Style Pictogram Generation

This pipeline trains a LoRA adapter on Stable Diffusion 1.5 to generate EasyRead-style pictograms for new concepts.

## Overview

The pipeline consists of:
1. Data preparation - Convert datasets into training format
2. LoRA training - Finetune SD 1.5 with LoRA adapters
3. Inference - Generate new pictograms using trained model

📖 **For detailed information about data preprocessing, see [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)**

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

## Checkpoint

Download our checkpoint (trained on LDS, openmoji and augmented ARASAAC) [here](https://huggingface.co/rllover123/easyread-dsl)

### Hardware Requirements

- GPU: Minimum 12GB VRAM (16GB+ recommended)
- RAM: 16GB+ system RAM
- Storage: ~5GB for model weights, ~1GB for dataset

## Usage: Generate Pictograms

Generate new pictograms using trained model:

```bash
python generate_pictograms.py \
    --lora_weights="path/to/checkpoint/checkpoint-final" \
    --prompt="a person on a rock with a blue shirt and a red hat; background color: yellow; skin color: black; hair color: blonde" \
    --output=path/to/output/person_on_rock.png" \
    --num_images=4 \
    --seed=42 \
    --instance_token="sks"
```

**Parameters:**
- `--lora_weights`: Path to trained LoRA weights (use checkpoint or final)
- `--prompt`: Description of pictogram to generate
- `--negative_prompt`: What to avoid (defaults work well for pictograms)
- `--num_images`: Generate multiple variations
- `--steps`: More steps = higher quality (30 is good, 50 for best)
- `--guidance_scale`: How closely to follow prompt (7.5 default, try 5-10)
- `--seed`: For reproducible results


For controllability please append the prompt with `; background color: {BACKGROUND_COLOR}; skin color: {SKIN_COLOR}; hair color: {HAIR_COLOR}`.
The training controllability parameters are the following (note that some of the terms are outdated but are used in ARASAAC):

SKIN_COLORS: white, black, assian, mulatto, aztec]  
HAIR_COLORS = blonde, brown, darkBrown, gray, darkGray, red, black  
BACKGROUND_COLORS = red, green, blue, yellow, black, white  


## Training

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

## File Structure

```
├── src/                                  # Source code
│   ├── dataset_creation/                 # Dataset collection scripts
│   │   ├── arasaac/                      # ARASAAC pictogram scraper
│   │   ├── icon645/                      # Icon645 dataset scripts
│   │   ├── lds/                          # LDS dataset scripts
│   │   ├── openmoji/                     # OpenMoji dataset scripts
│   │   └── quickdraw/                    # QuickDraw dataset scripts
│   ├── data_format_regularization/       # Data preparation and formatting
│   │   ├── prepare_dataset.py            # Prepare data for training
│   │   ├── add_prompts.py                # Add prompts to dataset
│   │   ├── summarize_dataset.py          # Dataset statistics
│   │   └── regularize_data_job.sh        # Batch processing script
│   ├── training/                         # Model training scripts
│   │   ├── generate_pictograms.py        # Inference script
│   └── evaluation/                       # Evaluation and metrics
│       ├── easyread_metrics.py           # EasyRead scoring metrics
│       └── easyread_analysis.py          # Analysis and visualization
├── data/                                 # Datasets and training data
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
