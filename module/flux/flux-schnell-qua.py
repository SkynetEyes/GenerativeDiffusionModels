import os
import torch
from diffusers import FluxPipeline

# *** CHANGE THIS LINE ***
# Import TorchAoConfig directly from diffusers (as documented)
from diffusers import TorchAoConfig as DiffusersTorchAoConfig

# Transformers TorchAoConfig can usually be imported from the root
from transformers import TorchAoConfig as TransformersTorchAoConfig

# This import path is ALREADY correct for the config wrapper
from diffusers.quantizers import PipelineQuantizationConfig

from huggingface_hub import login

# --- Login to Hugging Face ---
login("***REMOVED***")

# --- Fix CUDA fragmentation ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DISABLE_CACHING_ALLOCATOR_WARMUP"] = "1"
os.environ["DIFFUSERS_CUDA_CACHING_ALLOCATOR"] = "disable"

# --- Load FLUX model ---
model_id = "black-forest-labs/FLUX.1-schnell"


#pipeline_quant_config = PipelineQuantizationConfig(
#    quant_mapping={
#        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
#        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
#    }
#)


pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
         "transformer": DiffusersTorchAoConfig("int4_weight_only"),
         "text_encoder_2": TransformersTorchAoConfig("int4_weight_only"),
    }
)


pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="balanced",  # allowed by FluxPipeline
)

# --- Enable memory-saving features ---
#pipe.enable_model_cpu_offload()
#pipe.enable_sequential_cpu_offload() # aggressively moves layers to CPU
pipe.enable_attention_slicing()      # reduces attention VRAM load


print("[INFO] Model loaded")
print(pipe)
#count_parameters(pipe.transformer)
#measure_memory_usage("After model load")

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

#print("[INFO] Image generated")
#measure_memory_usage("After generation")

# --- Save image ---
image.save("flux-schnell-qua.png")
print("[DONE] Image saved")