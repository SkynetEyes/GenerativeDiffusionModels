import os
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


# ----------------------------
# Load SDXL Base with offload
# ----------------------------
print("Loading SDXL Base model...")
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
    variant="fp16",
    torch_dtype=torch.float16,
)

# Load IP-Adapter into base
print("Loading IP-Adapter weights...")
base.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.safetensors",
)
base.set_ip_adapter_scale(0.8)
base.enable_model_cpu_offload()

# ----------------------------
# Load Refiner with offload
# ----------------------------
print("Loading SDXL Refiner model...")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    use_safetensors=True,
    variant="fp16",
    torch_dtype=torch.float16,
)
refiner.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png"
)

# ----------------------------
# Inference settings
# ----------------------------
prompt = "A polar bear sitting in a chair drinking a milkshake, cinematic lighting, photo realistic"
negative_prompt = "deformed, ugly, low quality, bad anatomy, blur"
n_steps = 100
high_noise_frac = 0.8 # Base will generate 80% of the steps, Refiner the last 20%

# ----------------------------
# Run BASE
# ----------------------------
print("Starting BASE stage inference (generating latents)...")
with torch.no_grad():
    images = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=image, # pass the preprocessed CUDA tensor
        height=512,
        width=512,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        guidance_scale=5.0,
        guidance_rescale=0.7,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="latent",
    ).images

# ----------------------------
# Run REFINER stage
# ----------------------------
print("Starting REFINER stage inference (upscaling and refining)...")
with torch.no_grad():
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=images,
        height=512,
        width=512,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        guidance_scale=5.0,
        guidance_rescale=0.7,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="pil",
    ).images[0]

image.save("sdxl-refiner-img2img.png")  # Save to file