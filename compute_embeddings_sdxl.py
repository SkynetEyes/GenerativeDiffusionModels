#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from tqdm.auto import tqdm

from diffusers import StableDiffusionXLPipeline


MAX_SEQ_LENGTH = 77
OUTPUT_PATH = "embeddings.parquet"


def generate_image_hash(image):
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()


def load_flux_dev_pipeline():
    id = "stabilityai/stable-diffusion-xl-base-1.0"
    # Carregar pipeline SDXL para gerar embeddings
    pipeline = StableDiffusionXLPipeline.from_pretrained(id, device_map="balanced")
    return pipeline


@torch.no_grad()
def compute_embeddings(pipeline, prompts, max_sequence_length):
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    for prompt in tqdm(prompts, desc="Encoding prompts."):
        # SDXL: encode_prompt com output_hidden_states=True para obter pooled embeddings
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            device=pipeline.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,  # Ativar para obter tanto positive quanto negative
        )
        
        all_prompt_embeds.append(prompt_embeds)
        all_pooled_prompt_embeds.append(pooled_prompt_embeds)
        # Para SDXL, criamos um text_ids dummy (não é usado em SDXL como em FLUX)
        all_text_ids.append(torch.zeros(max_sequence_length, 1))

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"Max memory allocated: {max_memory:.3f} GB")
    return all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids


def run(args):
    dataset = load_dataset("Norod78/Yarn-art-style", split="train")
    image_prompts = {generate_image_hash(sample["image"]): sample["text"] for sample in dataset}
    all_prompts = list(image_prompts.values())
    print(f"{len(all_prompts)=}")

    pipeline = load_flux_dev_pipeline()
    all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids = compute_embeddings(
        pipeline, all_prompts, args.max_sequence_length
    )

    data = []
    for i, (image_hash, _) in enumerate(image_prompts.items()):
        data.append((image_hash, all_prompt_embeds[i], all_pooled_prompt_embeds[i], all_text_ids[i]))
    print(f"{len(data)=}")

    # Create a DataFrame
    embedding_cols = ["prompt_embeds", "pooled_prompt_embeds", "text_ids"]
    df = pd.DataFrame(data, columns=["image_hash"] + embedding_cols)
    print(f"{len(df)=}")

    # Convert embedding lists to arrays (for proper storage in parquet)
    def safe_convert(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            # Remove batch dimension if present
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            elif x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            return x.tolist()  # Converter para lista para parquet storage
        if isinstance(x, np.ndarray):
            # Remove batch dimension if present
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

    # Save the dataframe to a parquet file
    df.to_parquet(args.output_path)
    print(f"Data successfully serialized to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length to use for computing the embeddings. The more the higher computational costs.",
    )
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to serialize the parquet file.")
    args = parser.parse_args()

    run(args)
