import os
import torch
from diffusers import FluxPipeline

# Fix CUDA fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    device_map="balanced",
    load_in_4bit=True,   # <-- this is the key for 4-bit quantization
)

# --- Enable memory-saving features ---
pipe.enable_model_cpu_offload()      # offload full model when not used
pipe.enable_sequential_cpu_offload() # aggressively moves layers to CPU
pipe.enable_attention_slicing()      # reduces attention VRAM load

prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt,
    height=128,          # recommended for your GPU
    width=128,
    guidance_scale=1.5,  # much lighter than 3.5
    num_inference_steps=40,   # faster and lighter
    max_sequence_length=50,  # reduce text encoder memory
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("flux-dev.png")
