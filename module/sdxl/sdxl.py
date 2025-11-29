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
    output_folder: str
):
    print(f"\nRunning: prompt={prompt_label}, neg={use_negative}, "
          f"steps={n_steps}, size={height}x{width}, gs={guidance_scale}, gr={guidance_rescale}")

    # Load model
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    base.enable_model_cpu_offload()

    # Build call arguments
    call_args = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="pil",
    )

    # Add negative prompt if enabled
    if use_negative:
        call_args["negative_prompt"] = negative_prompt

    # Generate image
    image = base(**call_args).images[0]

    # Create folder path
    os.makedirs(output_folder, exist_ok=True)

    # Neg flag in filename
    neg_flag = "negON" if use_negative else "negOFF"

    # Filename
    filename = (
        f"sdxl_{prompt_label}_h{height}_w{width}_steps{n_steps}_"
        f"gs{guidance_scale}_gr{guidance_rescale}_{neg_flag}.png"
    )

    filepath = os.path.join(output_folder, filename)

    # Save
    image.save(filepath)
    print(f"Saved: {filepath}")


if __name__ == '__main__':
    STEPS = [50, 100]
    HEIGHT_WIDTH = [256, 512, 1024]
    GUIDANCE_SCALE = [3.0, 5.0, 7.0]
    GUIDANCE_RESCALE = [0.5, 0.7, 0.9]

    # Folder that will contain all difficulty groups
    ROOT_FOLDER = "sdxl_outputs"

    # Three difficulty prompts
    prompts = {
        "easy": "A red apple on a wooden table, soft natural lighting",
        "medium": "A futuristic motorcycle parked in a neon-lit alley, rainy reflections, cyberpunk style",
        "difficult": "A grand fantasy castle floating above the clouds, surrounded by flying ships, intricate architecture, warm sunset dramatic lighting"
    }

    negative_prompt = "deformed, ugly, low quality, bad anatomy, blur"

    # Loop over: difficulty × negON/negOFF × parameters
    for prompt_label, prompt_text in prompts.items():

        # Folder for this difficulty
        difficulty_folder = os.path.join(ROOT_FOLDER, prompt_label)

        for use_negative in [True, False]:

            for n_steps, size, guidance_scale, guidance_rescale in itertools.product(
                STEPS, HEIGHT_WIDTH, GUIDANCE_SCALE, GUIDANCE_RESCALE
            ):
                height = size
                width = size

                main(
                    prompt_label,
                    prompt_text,
                    negative_prompt,
                    use_negative,
                    n_steps,
                    height,
                    width,
                    guidance_scale,
                    guidance_rescale,
                    difficulty_folder
                )