#!/usr/bin/env python
# coding=utf-8
import argparse
import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

class SDImageDataset(Dataset):
    def __init__(self, embeddings_path, images_path, image_column="image_path", size=512, center_crop=False):
        self.embeddings_df = pd.read_parquet(embeddings_path)
        self.images_df = pd.read_parquet(images_path)
        self.size = size
        self.center_crop = center_crop
        # Merge to get image path for each embedding
        if image_column not in self.images_df.columns:
            raise ValueError(f"Coluna '{image_column}' não encontrada no parquet de imagens. Colunas disponíveis: {self.images_df.columns.tolist()}")
        # Realiza o merge e renomeia a coluna de imagem para evitar conflito
        merge_df = self.embeddings_df.merge(self.images_df[["image_hash", image_column]], on="image_hash", how="left", suffixes=("_emb", "_img"))
        # Se houver duplicidade, prioriza a coluna do parquet de imagens
        if f"{image_column}_img" in merge_df.columns:
            self.image_paths = merge_df[f"{image_column}_img"].tolist()
        else:
            self.image_paths = merge_df[image_column].tolist()
        self.df = merge_df
        self.prompt_embeds = self.df["prompt_embeds"].tolist()
        # pooled_prompt_embeds não existe para SD
        self.text_ids = self.df["text_ids"].tolist() if "text_ids" in self.df.columns else [None] * len(self.df)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.size, self.size), Image.LANCZOS)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        def to_tensor(x):
            arr = np.array(x)
            if arr.dtype == np.object_:
                arr = np.array([np.array(e, dtype=np.float32) for e in arr])
            arr = arr.astype(np.float32)
            return torch.from_numpy(arr)

        prompt_embeds = to_tensor(self.prompt_embeds[idx])
        text_ids = to_tensor(self.text_ids[idx])
        return {
            "pixel_values": img,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    prompt_embeds = torch.stack([b["prompt_embeds"] for b in batch])
    text_ids = torch.stack([b["text_ids"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "prompt_embeds": prompt_embeds,
        "text_ids": text_ids,
    }

def main(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[kwargs],
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    train_dataset = SDImageDataset(
        embeddings_path=args.embeddings_parquet,
        images_path=args.images_parquet,
        image_column=args.image_column,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    # Inicializa o tracker wandb se report_to for wandb
    if accelerator.is_main_process and args.report_to == "wandb":
        tracker_name = os.environ.get("WANDB_PROJECT", "huggingface")
        accelerator.init_trackers(tracker_name, config=vars(args))
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device)
                pixel_values = pixel_values.to(vae.device)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(model_input.device)
                sqrt_alpha_prod = torch.sqrt(alphas_cumprod[timesteps]).reshape(-1,1,1,1)
                sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod[timesteps]).reshape(-1,1,1,1)
                noisy_model_input = sqrt_alpha_prod * model_input + sqrt_one_minus_alpha_prod * noise
                noisy_model_input = noisy_model_input.to(accelerator.device)
                timesteps = timesteps.to(accelerator.device)
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                # Para SD, não há pooled_prompt_embeds nem added_cond_kwargs
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                model_pred = model_pred.to(accelerator.device)
                noise = noise.to(accelerator.device)
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
    # Salvar apenas os pesos LoRA do UNet (compatível com diffusers SD) ao final do treinamento
    unet.save_attn_procs(args.output_dir)
    print("Treinamento finalizado e LoRA salvo corretamente para SD.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--embeddings_parquet", type=str, required=True)
    parser.add_argument("--images_parquet", type=str, required=True)
    parser.add_argument("--image_column", type=str, default="image_path", help="Nome da coluna com os caminhos das imagens no parquet.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Número máximo de steps de treinamento. Se definido, ignora epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    main(args)
