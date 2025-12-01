#!/usr/bin/env python
# coding=utf-8
import argparse
import numpy as np
import pandas as pd
import torch
from huggingface_hub.utils import insecure_hashlib
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline

MAX_SEQ_LENGTH = 77
OUTPUT_PATH = "embeddings_sd.parquet"


def generate_image_hash(image):
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()


def load_sd_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline

@torch.no_grad()
def compute_embeddings(pipeline, prompts, max_sequence_length):
    all_prompt_embeds = []
    all_text_ids = []
    for prompt in tqdm(prompts, desc="Encoding prompts."):
        # SD: encode prompt using text_encoder
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(pipeline.device)
        prompt_embeds = pipeline.text_encoder(input_ids)[0]  # [1, 77, 768]
        all_prompt_embeds.append(prompt_embeds.squeeze(0).cpu().numpy())
        all_text_ids.append(input_ids.squeeze(0).cpu().numpy())
    return all_prompt_embeds, all_text_ids


def run(args):
    df_in = pd.read_parquet(args.input_parquet)
    print(f"Carregado parquet: {args.input_parquet} ({len(df_in)} linhas)")
    if "text" in df_in.columns:
        all_prompts = df_in["text"].tolist()
    elif "prompt" in df_in.columns:
        all_prompts = df_in["prompt"].tolist()
    elif "label" in df_in.columns:
        all_prompts = df_in["label"].tolist()
    else:
        raise ValueError("O parquet precisa ter uma coluna 'text' ou 'prompt'.")
    image_hashes = df_in["image_hash"].tolist() if "image_hash" in df_in.columns else list(range(len(df_in)))
    print(f"{len(all_prompts)=}")
    pipeline = load_sd_pipeline()
    all_prompt_embeds, all_text_ids = compute_embeddings(
        pipeline, all_prompts, args.max_sequence_length
    )
    data = []
    for i, image_hash in enumerate(image_hashes):
        data.append((image_hash, all_prompt_embeds[i], all_text_ids[i]))
    print(f"{len(data)=}")
    embedding_cols = ["prompt_embeds", "text_ids"]
    df = pd.DataFrame(data, columns=["image_hash"] + embedding_cols)
    print(f"{len(df)=}")
    def safe_convert(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            elif x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            return x.tolist()
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            elif x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            return x.tolist()
        if isinstance(x, list):
            return x
        return x
    for col in embedding_cols:
        df[col] = df[col].apply(safe_convert)
    df.to_parquet(args.output_path)
    print(f"Data successfully serialized to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True, help="Arquivo parquet de entrada com prompts.")
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    run(args)
