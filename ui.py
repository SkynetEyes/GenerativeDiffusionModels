# app.py
from scipy import io
import streamlit as st
import torch
from diffusers import BitsAndBytesConfig, UNet2DConditionModel
from diffusers import BitsAndBytesConfig
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector, PidiNetDetector, LineartDetector
from typing import List, Dict, Any
import os
import io


# ---------------- CONFIG ----------------
CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# Se você tiver pesos LoRA local (opcional)
LORA_CONFIGS = {
    'Yarn':   { 'path': 'sdxl_yarn',      'filename': 'pytorch_lora_weights.safetensors' },
    'Rembrandt portrait': { 'path': 'rembrandt_portrait_sdxl', 'filename': 'pytorch_lora_weights.safetensors' },
    'Rembrandt barroco': { 'path': 'rembrandt_barroco_sdxl', 'filename': 'pytorch_lora_weights.safetensors' },
    'Van Gogh portrait':  { 'path': 'vangogh_portrait',           'filename': 'pytorch_lora_weights.safetensors' },
    'Van Gogh style':    { 'path': 'vangogh_style',              'filename': 'pytorch_lora_weights.safetensors' },
}

# Detect device
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


ADAPTER_TYPES = {
    "Canny": {
        "model": "TencentARC/t2i-adapter-canny-sdxl-1.0",
        "pre_factory": lambda: CannyDetector(),
    },
    "Depth": {
        "model": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
        "pre_factory": lambda: (MidasDetector.from_pretrained("lllyasviel/Annotators")),
    },
    "OpenPose":{
        "model": "TencentARC/t2i-adapter-openpose-sdxl-1.0",
        "pre_factory": lambda: OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
    },
    "Sketch":{
        "model": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        "pre_factory": lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
    },
    'Lineart': {
        "model": "TencentARC/t2i-adapter-lineart-sdxl-1.0",
        "pre_factory": lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"),
    }
    
}

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(page_title="Diffusion ControlNet UI (otimizado)", layout="wide")
st.title("Arte Controlada e Estilo — Versão Otimizada")

# ----------------- SESSION STATE -----------------
if "selected_controls" not in st.session_state:
    st.session_state.selected_controls = set()
if "selected_style" not in st.session_state:
    st.session_state.selected_style = None
if "ref_image" not in st.session_state:
    st.session_state.ref_image = None
if "preprocessors" not in st.session_state:
    st.session_state.preprocessors = {}  # cache dos preprocessors (detectors)

# ----------------- HELPERS: carregamento cacheado das pipelines -----------------

def load_adapter_models(adapter_list: List[str], torch_dtype=torch.float16):
    """
    Carrega modelos T2I-Adapter para a lista de nomes (adapter_list).
    Retorna lista de T2IAdapter (na mesma ordem).
    """
    adapters = []
    for adp in adapter_list:
        cfg = ADAPTER_TYPES.get(adp)
        if cfg is None:
            continue
        model_id = cfg["model"]
        model = T2IAdapter.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        adapters.append(model)
    return adapters


def load_pipeline_with_adapter(base_model: str, adapter_list: List[str], torch_dtype=torch.float16):
    # Configuração de quantização para o UNet
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    """
    Cria e retorna uma StableDiffusionAdapterPipeline com os adapters carregados.
    """
    # Inspirado no exemplo fornecido
    adapter = None
    if adapter_list:
        # Carrega adapter SEM quantização
        cfg = ADAPTER_TYPES.get(adapter_list[0])
        if cfg:
            model_id = cfg["model"]
            adapter = T2IAdapter.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(DEVICE)
    model_id = base_model
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, quant_config=quant_config)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
        quantization_config=quant_config
    )
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id,
        vae=vae,
        unet=unet,
        adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    ).to(DEVICE)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def try_load_lora_weights(pipe, path: str, filename: str):
    """Tenta carregar pesos LoRA local (opcional). Não falha se não existir."""
    full = os.path.join(path, filename)
    if os.path.exists(full):
        try:
            pipe.load_lora_weights(path, weight_name=filename)
            return True
        except Exception as e:
            st.warning(f"Falha ao carregar LoRA: {e}")
            return False
    return False

# ----------------- UI: escolha de controles e upload/desenho -----------------
def adapter_input_ui():
    st.markdown("### Adapter Type")
    st.markdown("---")
    adapter_types_ui = ["ALL"] + list(ADAPTER_TYPES.keys())

    col1, _, col2 = st.columns([0.35, 0.02, 0.63])
    with col1:
        for ct in adapter_types_ui:
            is_selected = ct in st.session_state.selected_controls
            clicked = st.button(
                ct,
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                key=f"btn_{ct}"
            )
            if clicked:
                if ct == "ALL":
                    all_set = set(adapter_types_ui)
                    all_set.discard("ALL")
                    if st.session_state.selected_controls >= all_set:
                        st.session_state.selected_controls = set()
                    else:
                        st.session_state.selected_controls = set(all_set)
                    st.rerun()
                else:
                    if ct in st.session_state.selected_controls:
                        st.session_state.selected_controls.remove(ct)
                    else:
                        st.session_state.selected_controls.add(ct)
                    st.rerun()

    with col2:
        st.markdown("### Selecione o método de entrada")
        input_mode = st.radio("Como deseja fornecer a imagem base?", ["Upload", "Desenhar"], horizontal=True)

        img = None

        if input_mode == "Upload":
            uploaded = st.file_uploader(label='Imagem de Controle', type=["jpg", "jpeg", "png"])
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Imagem enviada", width=400)
                st.session_state.ref_image = np.array(img)

        else:  # Desenhar
            st.markdown("Desenhe sua imagem abaixo:")
            c1, c2 = st.columns(2)
            with c1:
                from streamlit_drawable_canvas import st_canvas
                canvas = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=4,
                    stroke_color="#000000",
                    background_color="#ffffff",
                    height=400,
                    width=400,
                    drawing_mode="freedraw",
                    key="canvas",
                )
            with c2:
                if canvas.image_data is not None:
                    img = canvas.image_data.astype("uint8")
                    st.image(img, caption="Desenho capturado", width=400)
                    st.session_state.ref_image = img

# ----------------- UI: styles (simples) -----------------
def styles_ui():
    st.markdown("---")
    st.markdown("### Styles")
    cols = st.columns(3)

    def render_button(label, col, key):
        clicked = col.button(
            label,
            use_container_width=True,
            type="primary" if st.session_state.selected_style == label else "secondary",
            key=key
        )
        if clicked:
            st.session_state.selected_style = label
            st.rerun()

    for i, style_name in enumerate(['Yarn', 'Rembrandt portrait','Rembrandt barroco', 'Van Gogh portrait', 'Van Gogh style']):
        render_button(style_name, cols[i % 3], f"style_{i}")

    return st.session_state.selected_style

def size_selector_ui():
    st.markdown("---")
    st.markdown("### Tamanho da Imagem")
    size_options = [256, 512, 768, 1024]
    selected_size = st.selectbox("Selecione o tamanho da imagem gerada:", size_options, index=1)
    return selected_size

# ----------------- Preprocess images (lazy detectors) -----------------
def get_preprocessor(name: str):
    """Retorna (e cria se necessario) o preprocessor para 'name' e cacheia em session_state."""
    if name in st.session_state.preprocessors:
        return st.session_state.preprocessors[name]
    cfg = ADAPTER_TYPES.get(name)
    if cfg is None:
        return None
    factory = cfg.get("pre_factory")
    if factory is None:
        return None
    proc = factory()
    st.session_state.preprocessors[name] = proc
    return proc

def sidebar_preprocess_images(ref_image, selected_controls):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Ajustes de Geração")
    inference_steps = st.sidebar.slider("Número de passos de inferência", min_value=10, max_value=100, value=30, step=5)
    guidance_scale = st.sidebar.slider("Escala de orientação", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    adapter_conditioning_scale = st.sidebar.slider("Escala de condicionamento do adapter", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    adapter_conditioning_factor = st.sidebar.slider("Fator de condicionamento do adapter", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    return inference_steps, guidance_scale, adapter_conditioning_scale, adapter_conditioning_factor

def preprocess_images(ref_image, selected_controls):
    """
    Retorna lista de PIL.Image (RGB), uma por control selecionado, na mesma ordem de selected_controls.
    Garante que imagens 2D -> RGB (3 canais).
    """
    outputs = []
    if ref_image is None:
        return outputs

    # se ref_image estiver como numpy array HxW ou HxWxC, garantimos tipo uint8
    if isinstance(ref_image, np.ndarray):
        base_img = Image.fromarray(ref_image.astype("uint8")).convert("RGB")
    elif isinstance(ref_image, Image.Image):
        base_img = ref_image.convert("RGB")
    else:
        # fallback - tenta converter
        base_img = Image.fromarray(np.array(ref_image).astype("uint8")).convert("RGB")

    for ctl in selected_controls:
        if ctl not in ADAPTER_TYPES:
            continue
        proc = get_preprocessor(ctl)
        if proc is None:
            continue
        cond = proc(np.array(base_img))  # muitos detectors aceitam np.array input
        # cond pode ser:
        # - PIL.Image
        # - numpy array HxW (mask) ou HxWx1 ou HxWx3
        if isinstance(cond, Image.Image):
            cond_img = cond.convert("RGB")
        else:
            cond_arr = np.asarray(cond)
            # Se for 2D (H,W) -> replicar canais para RGB
            if cond_arr.ndim == 2:
                cond_arr = np.stack([cond_arr, cond_arr, cond_arr], axis=-1)
            # Se for HxWx1 -> squeeze e replicar
            if cond_arr.ndim == 3 and cond_arr.shape[2] == 1:
                cond_arr = np.concatenate([cond_arr]*3, axis=2)
            # Assegura uint8
            cond_arr = cond_arr.astype("uint8")
            cond_img = Image.fromarray(cond_arr).convert("RGB")
    # Redimensiona imagem filtrada para 512x512 antes de mostrar
    cond_img = cond_img.resize((512, 512))
    outputs.append(cond_img)

    return outputs
# ----------------- Geração única (pipeline cacheada por lista de controlnets) -----------------
def generate_image(prompt: str, cond_images: List[np.ndarray], selected_controls: List[str], selected_size: int, seed: int = 0, num_inference_steps: int = 30, guidance_scale: float = 7.5, adapter_conditioning_scale: float = 0.8, adapter_conditioning_factor: float = 1.0):
    # Adiciona sufixo de estilo ao prompt conforme seleção
    style_suffixes = {
        'Yarn': ', yarn art style.',
        'Rembrandt portrait': ', artstyle rembrandt.',
        'Rembrandt barroco': ', artstyle rembrandt.',
        'Van Gogh style': ', artstyle vangogh.',
        'Van Gogh portrait': ', artstyle vangogh.'
    }
    selected_style = st.session_state.get('selected_style', None)
    if selected_style in style_suffixes:
        prompt = prompt.strip() + style_suffixes[selected_style]
    # Garantir que image_arg está definido antes de acessar shape
    if not cond_images or len(cond_images) == 0:
        st.error("Você precisa fornecer uma imagem de referência para geração.")
        return None
    else:
        # Garante que image_arg é PIL.Image
        if isinstance(cond_images[0], np.ndarray):
            image_arg = Image.fromarray(cond_images[0].astype("uint8")).convert("RGB")
        elif isinstance(cond_images[0], Image.Image):
            image_arg = cond_images[0].convert("RGB")
        else:
            image_arg = Image.fromarray(np.array(cond_images[0]).astype("uint8")).convert("RGB")
    height = image_arg.height
    width = image_arg.width
    """
    Gera uma imagem usando pipeline cacheada (dependente do conjunto de controlnets).
    Retorna PIL Image.
    """
    # escolher dtype: se GPU, float16; caso contrário float32
    torch_dtype = torch.float16 if CUDA_AVAILABLE else torch.float32

    # Carrega pipeline (cacheada por control_list)
    pipe = load_pipeline_with_adapter(CKPT_ID, selected_controls, torch_dtype=torch_dtype)
    print('ok')
    # Seleciona LoRA conforme estilo
    lora_path = None
    lora_filename = None
    selected_style = st.session_state.get('selected_style', None)
    if selected_style and selected_style in LORA_CONFIGS:
        lora_path = LORA_CONFIGS[selected_style]['path']
        lora_filename = LORA_CONFIGS[selected_style]['filename']
        try_load_lora_weights(pipe, lora_path, lora_filename)

    # mover para dispositivo (a load_pipeline já tem offload; chamar .to pode forçar toda a pipeline na GPU -> evitar)
    # pipe.to(DEVICE)  # geralmente não necessário se enable_model_cpu_offload foi ativado

    generator = None
    if CUDA_AVAILABLE:
        generator = torch.Generator("cuda").manual_seed(0)
    else:
        generator = torch.Generator("cpu").manual_seed(0)
    print(len(cond_images))
    print(selected_controls)
    # Executa
    # image_arg já é PIL.Image, não redefinir para numpy
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image_arg,
            height=selected_size,
            width=selected_size,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            adapter_conditioning_factor=adapter_conditioning_factor,
            generator=generator,
        )
    # resultado é um objeto PipelineOutput; images disponível em .images
    out_img = result.images[0]

    # cleanup
    try:
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out_img

# ----------------- LAYOUT PRINCIPAL -----------------
st.sidebar.markdown("## Configurações")
st.sidebar.write(f"Device: **{DEVICE}**")
if CUDA_AVAILABLE:
    st.sidebar.write("CUDA disponível — usando optimizações (offload, attention slicing quando possível).")
else:
    st.sidebar.write("Sem CUDA — usando CPU (mais lento).")

# ControlNet + input
adapter_input_ui()

# Styles
_ = styles_ui()

# Render size selector once in main layout
selected_size = size_selector_ui()

inference_steps, guidance_scale, adapter_conditioning_scale, adapter_conditioning_factor = sidebar_preprocess_images(st.session_state.ref_image, list(st.session_state.selected_controls))

st.markdown("---")
st.markdown("### Prompt")
prompt = st.text_input(label="Prompt", value="A fantasy portrait, cinematic lighting")

# Exibe imagens condicionais pré-processadas (se houver)
if st.session_state.ref_image is not None and len(st.session_state.selected_controls) > 0:
    img = st.session_state.ref_image
    control_images_preview = {}
    for ctrl in st.session_state.selected_controls:
        if ctrl in ADAPTER_TYPES:
            proc = get_preprocessor(ctrl)
            if proc is None:
                continue
            processed = proc(img)
            control_images_preview[ctrl] = processed
            st.image(processed, caption=f"Adapter: {ctrl}", width=240)

# Botão único para gerar
if st.button("Gerar imagem", type="primary"):
    if st.session_state.ref_image is None:
        st.error("Envie ou desenhe uma imagem de referência antes de gerar.")
        st.stop()

    if len(st.session_state.selected_controls) == 0:
        st.error("Selecione ao menos um ControlNet para condicionar a geração.")
        st.stop()

    with st.spinner("Pré-processando imagens..."):
        cond_images = preprocess_images(st.session_state.ref_image, list(st.session_state.selected_controls))

    with st.spinner("Gerando imagem (isso usa GPU/CPU e pode levar alguns segundos)..."):
        try:
            final = generate_image(
                prompt,
                cond_images,
                list(st.session_state.selected_controls),
                selected_size,
                seed=0,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                adapter_conditioning_scale=adapter_conditioning_scale,
                adapter_conditioning_factor=adapter_conditioning_factor
            )
            st.image(final, caption="Imagem Final", width=selected_size)
            img_bytes = io.BytesIO()
            final.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Botão de download
            st.download_button(
                label="Baixar imagem",
                data=img_bytes,
                file_name="imagem_gerada.png",
                mime="image/png"
            )
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            st.error(f"Erro durante a geração: {e}")
            # tenta liberar memória
            try:
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
