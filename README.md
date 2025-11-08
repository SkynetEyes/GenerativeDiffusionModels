# GenerativeDiffusionModels

## Objective
This project have as objective generate images with specific art style using Adapters like LoRa and ControlNet

## How run
```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requeriments.txt
```
Generate the text embeddings using the compute_embeddigs.py
```bash
python compute_embeddings.py \
  --max_sequence_length 77 \
  --output_path embeddings.parquet
```
Run the training with accelerate
```bash
accelerate launch 
  train.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --data_df_path="embeddings.parquet" \
  --output_dir="results" \
  --mixed_precision="bf16" \
  --use_8bit_adam \
  --weighting_scheme="none" \
  --width=512 \
  --height=768 \
  --train_batch_size=1 \
  --repeats=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --report_to="wandb" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \ 
  --lr_scheduler="constant" \
  --checkpointing_steps=100 \
  --lr_warmup_steps=0 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=700 \
  --seed="0"
```