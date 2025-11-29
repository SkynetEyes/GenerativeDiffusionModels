import os
import torch
import psutil
from diffusers import FluxPipeline
from huggingface_hub import login


# --- Login to Hugging Face ---
login("***REMOVED***")

# --- Fix CUDA fragmentation ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- Load FLUX model ---
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    device_map="balanced",   # allowed by FluxPipeline
)
ATTENTION_SLICING = pipe.enable_attention_slicing()

# --- Enable memory-saving features ---
#pipe.enable_model_cpu_offload()
#pipe.enable_sequential_cpu_offload() # aggressively moves layers to CPU

print("[INFO] Model loaded")
print(pipe)

# --- Generate image ---
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=256,
    width=256,
    guidance_scale=0.0,
    num_inference_steps=25,
    max_sequence_length=100,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# --- Save image ---
image.save("flux-schnell.png")
print("[DONE] Image saved")
