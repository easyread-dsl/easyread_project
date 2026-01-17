# Data Preprocessing Pipeline

This document describes the complete preprocessing pipeline that data goes through in the EasyRead pictogram generation system, from raw datasets to training-ready inputs.

## Overview

The preprocessing pipeline consists of three main stages:

1. **Dataset Preparation & Format Regularization** - Consolidate multiple datasets into a unified format
2. **Prompt Generation** - Generate descriptive captions using BLIP-2 vision-language model
3. **Training-Time Preprocessing** - Real-time transformations during model training

---

## Stage 1: Dataset Preparation & Format Regularization

**Script:** `src/data_format_regularization/prepare_dataset.py`

This stage integrates multiple icon/pictogram datasets (ARASAAC, AAC, Mulberry, Icon645, LDS, OpenMoji) into a unified `training_data/` directory with standardized naming and metadata.

### Input Sources

The pipeline supports multiple pictogram datasets:
- **ARASAAC**: Large pictogram dataset with extensive metadata
- **AAC** (AACIL): Alternative and Augmentative Communication symbols
- **Mulberry**: Open-source pictogram set (SVG format)
- **Icon645**: Standardized icon collection
- **LDS**: Line drawing set
- **OpenMoji**: Open-source emoji-style pictograms

### Processing Steps

#### 1.1 File Consolidation & Naming

Each dataset is processed to create consistent filenames:

```
{dataset}_{subfolder}_{original_name}.{ext}
```

**Examples:**
- `arasaac_animals_cats_cat.png`
- `aac_food_apple.png`
- `mulberry_Afraid_Man_3132.png`

**Key behaviors:**
- Loads existing `metadata.json` and merges new entries (no duplicates)
- Deduplicates by `(dataset, image_file)` tuple
- Avoids filename collisions via auto-renaming (`_1`, `_2`, etc.)
- Includes source subfolder names to prevent collisions

#### 1.2 Format Conversion

**SVG to PNG conversion** (for Mulberry and other SVG datasets):
- Target size: 256×256 pixels
- Conversion tools (in priority order):
  1. `cairosvg` (Python library)
  2. `rsvg-convert` (librsvg)
  3. `inkscape`

#### 1.3 Metadata Extraction & Normalization

For each image, the following metadata is extracted and standardized:

```json
{
  "dataset": "arasaac",
  "image_file": "arasaac_animals_cat_12345.png",
  "id": "12345",
  "title": "cat",
  "keywords": ["cat", "animal", "pet", "feline"],
  "categories": ["animals", "pets"],
  "license": "CC BY-NC-SA 4.0",
  "skin_color": "white",
  "hair_color": "blonde",
  "background_color": "yellow"
}
```

**Metadata fields:**
- **dataset**: Source dataset identifier
- **image_file**: Standardized filename in `training_data/images/`
- **id**: Original image ID from source dataset
- **title**: Primary label/title
- **keywords**: List of related keywords (parsed from pipe-separated strings)
- **categories**: Semantic categories (parsed from pipe-separated strings)
- **license**: License information
- **skin_color**, **hair_color**, **background_color**: Controllability parameters (ARASAAC-specific)

#### 1.4 Output Structure

```
training_data/
├── images/              # All images with standardized names
│   ├── arasaac_*.png
│   ├── aac_*.png
│   ├── mulberry_*.png
│   └── ...
├── metadata.json        # Complete metadata (array of objects)
└── metadata.csv         # Same data in CSV format
```

---

## Stage 2: Prompt Generation

**Script:** `src/data_format_regularization/add_prompts.py`

This stage generates high-quality textual descriptions (prompts) for each image using a vision-language model. These prompts are crucial for training the diffusion model to understand the content of pictograms.

### Model Used

**BLIP-2 FLAN-T5 XL** (`Salesforce/blip2-flan-t5-xl`)
- State-of-the-art vision-language model
- Generates semantic, content-focused descriptions
- Runs on GPU for optimal performance

### Processing Steps

#### 2.1 Image Loading

- Supports both raster formats (PNG, JPG, etc.) and SVG
- SVG files are rasterized using `cairosvg` if available
- All images converted to RGB mode

#### 2.2 Hint Building

The model is provided with contextual hints from existing metadata:

```python
hint = "cat, animal, pet, feline, animals, pets"  # from title, keywords, categories
```

These hints help guide the caption generation to be more accurate.

#### 2.3 Caption Generation

The model receives an instruction-based prompt:

```
"Describe concisely what this image depicts. 
Only mention the semantic content (objects, people, actions). 
Do not mention style, quality, or rendering. 
Use these hints only if they match the image: {hint}"
```

**Generation parameters:**
- `max_new_tokens`: 40
- `num_beams`: 4 (beam search for quality)
- `length_penalty`: 0.0
- `early_stopping`: True

**Example outputs:**
- Input: ARASAAC pictogram of a cat
- Hint: "cat, animal, pet"
- Output: "a cat sitting down with a collar"

#### 2.4 Batched Processing

- Processes in batches of 50 images
- Saves progress after each batch
- Creates backups of existing metadata
- Skips images already captioned by the same model

#### 2.5 Metadata Update

Each entry is updated with:
```json
{
  "prompt": "a cat sitting down with a collar",
  "prompt_model": "local:Salesforce/blip2-flan-t5-xl"
}
```

The `prompt_model` field tracks which model generated the caption, enabling:
- Re-captioning with better models without re-processing all images
- Quality control and comparison

---

## Stage 3: Training-Time Preprocessing

**Script:** `src/training/train_lora.py`

This stage applies real-time transformations when loading data during model training.

### Data Loading Process

#### 3.1 Metadata Loading

Supports two formats:
- **metadata.jsonl**: Line-delimited JSON (one object per line)
- **metadata.json**: JSON array

Field normalization:
- Accepts multiple field name variants: `file_name`, `image_file`, `image`, `path`
- Handles both string and list formats for keywords/categories
- Pipe-separated strings (`"cat|animal|pet"`) are split into lists

#### 3.2 Caption Construction

The training caption is built in a specific format:

```
{instance_token} {title}: {prompt}; background color: {bg_color}; skin color: {skin_color}; hair color: {hair_color}
```

**Example:**
```
sks cat: a cat sitting down with a collar; background color: yellow; skin color: white; hair color: blonde
```

**Components:**
- **instance_token** (`sks`): Special token to identify the style being learned
- **title**: Primary object label
- **prompt**: BLIP-2 generated description
- **color attributes**: Controllability parameters (if all three are present)

**Color vocabularies:**
- Skin colors: `white`, `black`, `assian`, `mulatto`, `aztec` (Note: "assian" is the spelling used in ARASAAC dataset)
- Hair colors: `blonde`, `brown`, `darkBrown`, `gray`, `darkGray`, `red`, `black`
- Background colors: `red`, `green`, `blue`, `yellow`, `black`, `white`

#### 3.3 Image Preprocessing

**Steps:**
1. Load image as RGB
2. Resize to 512×512 using Lanczos resampling (high-quality)
3. Normalize pixel values from [0, 255] to [-1, 1]:
   ```python
   image = (image / 127.5) - 1.0
   ```
4. Convert from HWC (Height, Width, Channels) to CHW format (PyTorch standard)

**No augmentation** is applied - images are used as-is to preserve the pictogram style characteristics.

#### 3.4 Text Preprocessing

**Tokenization:**
- Uses CLIP tokenizer from Stable Diffusion 1.5
- Maximum length: 77 tokens (model_max_length)
- Padding: Sequences padded to max length
- Truncation: Longer captions truncated

**Output:**
```python
{
    "pixel_values": tensor([-1.0 to 1.0], shape=[3, 512, 512]),
    "input_ids": tensor([...], shape=[77])
}
```

---

## Summary of Transformations

### Image Transformations

1. **Format conversion**: SVG → PNG (256×256)
2. **Resize**: Any size → 512×512 (Lanczos)
3. **Color space**: Any → RGB
4. **Normalization**: [0, 255] → [-1, 1]
5. **Tensor format**: HWC → CHW

### Text Transformations

1. **Metadata extraction**: Title, keywords, categories from source
2. **Caption generation**: BLIP-2 visual understanding
3. **Prompt construction**: Template with instance token and color attributes
4. **Tokenization**: Text → token IDs (CLIP tokenizer, max 77 tokens)

### Key Design Decisions

- **No data augmentation**: Preserves pictogram style consistency
- **Merge-safe processing**: Can re-run without duplicating data
- **Batch processing with checkpoints**: Handles large datasets reliably
- **Flexible metadata formats**: Supports multiple input structures
- **Controllability**: Color attributes enable fine-grained generation control

---

## Running the Pipeline

All commands should be run from the project root directory unless otherwise specified.

### Step 1: Prepare Dataset
```bash
python src/data_format_regularization/prepare_dataset.py
```
Output: `data/training_data/` with images and initial metadata

### Step 2: Generate Prompts
```bash
python src/data_format_regularization/add_prompts.py
```
Output: Updated metadata.json with BLIP-2 generated prompts

### Step 3: Train Model
```bash
python src/training/train_lora.py \
    --data_dir="data/training_data" \
    --output_dir="./lora_output"
```

---

## Quality Checks

Use the summary script to verify preprocessing:
```bash
python src/data_format_regularization/summarize_dataset.py
```

This provides:
- Total entries per dataset
- Prompt coverage statistics
- Prompt length distribution
- License information
- Category distribution
- Duplicate detection

---

## Common Issues & Solutions

**Issue**: SVG conversion fails
- **Solution**: Install one of: `pip install cairosvg`, `apt install librsvg2-bin`, or `apt install inkscape`

**Issue**: Out of memory during prompt generation
- **Solution**: Reduce batch size in `add_prompts.py` (default: 50)

**Issue**: Missing color attributes
- **Solution**: This is expected for non-ARASAAC datasets; training will proceed without color control

**Issue**: Prompt already exists but need to regenerate
- **Solution**: Either delete the `prompt` field from metadata.json, or change `PROMPT_MODEL_TAG` in `add_prompts.py`
