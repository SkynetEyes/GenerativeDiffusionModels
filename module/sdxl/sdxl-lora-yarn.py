import itertools
import os
from diffusers import DiffusionPipeline
import torch


def main(
    prompt_label: str,
    prompt: str,
    negative_prompt: str,
    use_negative: bool,
    n_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    guidance_rescale: float,
    output_folder: str,
):
    print(f"\nRunning: prompt={prompt_label}, neg={use_negative}, "
          f"steps={n_steps}, size={height}x{width}, gs={guidance_scale}, "
          f"gr={guidance_rescale}")

    # -------------------------------------------------------
    # Load SDXL Base + LoRA (NO REFINER)
    # -------------------------------------------------------
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
    )

    base.load_lora_weights(
        "/home/zionsflow/Documents/GenerativeDiffusionModels/lora-trained-xl-yarn-style/checkpoint-500",
        weight_name="pytorch_lora_weights.safetensors",
    )

    base.enable_model_cpu_offload()

    # -------------------------------------------------------
    # Build BASE call arguments (direct to PIL output)
    # -------------------------------------------------------
    base_args = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="pil",
    )

    if use_negative:
        base_args["negative_prompt"] = negative_prompt

    # -------------------------------------------------------
    # BASE stage (final pixels directly)
    # -------------------------------------------------------
    print("BASE stage running...")
    with torch.no_grad():
        final_image = base(**base_args).images[0]

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)

    neg_flag = "negON" if use_negative else "negOFF"

    filename = (
        f"sdxl_lora_{prompt_label}_h{height}_w{width}_steps{n_steps}_"
        f"gs{guidance_scale}_gr{guidance_rescale}_{neg_flag}.png"
    )

    filepath = os.path.join(output_folder, filename)
    final_image.save(filepath)
    print(f"Saved: {filepath}")


# ======================================================================
# MAIN LOOP — identical to your reference script
# ======================================================================
if __name__ == '__main__':

    STEPS = [50, 100]
    HEIGHT_WIDTH = [256, 512, 1024]
    GUIDANCE_SCALE = [3.0, 5.0, 7.0]
    GUIDANCE_RESCALE = [0.5, 0.7, 0.9]

    ROOT_FOLDER = "sdxl_lora_outputs"

    prompts = {
        "easy": "A red apple on a wooden table, soft natural lighting (yarn style)",
        "medium": "A futuristic motorcycle parked in a neon-lit alley, rainy reflections, cyberpunk style (yarn style)",
        "difficult": "A grand fantasy castle floating above the clouds, surrounded by flying ships, intricate architecture, warm sunset dramatic lighting (yarn style)"
    }

    negative_prompt = "deformed, ugly, low quality, bad anatomy, blur"

    # Loop: difficulty × negON/negOFF × all parameter combinations
    for prompt_label, prompt_text in prompts.items():

        difficulty_folder = os.path.join(ROOT_FOLDER, prompt_label)

        for use_negative in [True, False]:

            for n_steps, size, guidance_scale, guidance_rescale in itertools.product(
                STEPS, HEIGHT_WIDTH, GUIDANCE_SCALE, GUIDANCE_RESCALE
            ):
                main(
                    prompt_label,
                    prompt_text,
                    negative_prompt,
                    use_negative,
                    n_steps,
                    size,
                    size,
                    guidance_scale,
                    guidance_rescale,
                    difficulty_folder,
                )
