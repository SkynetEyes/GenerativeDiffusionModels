import os
import torch
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

# Fix CUDA fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id = "black-forest-labs/FLUX.1-dev"

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
pipe.to('cpu')

# --- Enable memory-saving features ---
pipe.enable_sequential_cpu_offload() # aggressively moves layers to CPU
pipe.enable_attention_slicing()      # reduces attention VRAM load

prompt = "A cat holding a sign that says hello world"

pipe_kwargs = {
    "prompt": prompt,
    "height": 128,
    "width": 128,
    "guidance_scale": 1,
    "num_inference_steps": 25,
    "max_sequence_length": 75,
}


print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

image = pipe(
    **pipe_kwargs, generator=torch.manual_seed(0),
).images[0]

print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

image.save("flux-dev_bnb_4bit.png")