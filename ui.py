# app.py
import streamlit as st
import torch
import gc
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from controlnet_aux import CannyDetector, OpenposeDetector, ZoeDetector, LineartDetector
from typing import List, Dict, Any
import os

# ---------------- CONFIG ----------------
CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# Se você tiver pesos LoRA local (opcional)
LORA_PATH = "pesos"
LORA_FILENAME = "pytorch_lora_weights_rem_por.safetensors"

# Detect device
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


CONTROL_TYPES = {
    "Canny": {
        "model": "lllyasviel/sd-controlnet-canny",
        "pre_factory": lambda: CannyDetector(),
    },
    "Depth": {
        "model": "lllyasviel/control_v11f1p_sd15_depth",
        "pre_factory": lambda: ZoeDetector.from_pretrained("lllyasviel/Annotators"),
    },
    "Pose": {
        "model": "xinsir/controlnet-openpose-sdxl-1.0",
        "pre_factory": lambda: OpenposeDetector.from_pretrained("lllyasviel/ControlNet"),
    },
    "Segmentation": {
        "model": "lllyasviel/control_v11p_sd15_lineart",
        "pre_factory": lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"),
    },
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
@st.cache_resource(show_spinner=False)
def load_controlnet_models(control_list: List[str], torch_dtype=torch.float16):
    """
    Carrega modelos ControlNet para a lista de nomes (control_list).
    Retorna lista de ControlNetModel (na mesma ordem).
    """
    controlnets = []
    for ctl in control_list:
        cfg = CONTROL_TYPES.get(ctl)
        if cfg is None:
            continue
        model_id = cfg["model"]
        model = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        controlnets.append(model)
    return controlnets

@st.cache_resource(show_spinner=False)
def load_pipeline_with_controlnet(base_model: str, control_list: List[str], torch_dtype=torch.float16):
    """
    Cria e retorna uma StableDiffusionControlNetPipeline com os controlnets carregados.
    A função é cacheada para não recarregar modelos repetidamente.
    """
    # Carrega controlnets (pode ser lista vazia)
    controlnets = load_controlnet_models(control_list, torch_dtype=torch_dtype) if control_list else None
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnets if controlnets else None,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )

    # Configurações que ajudam a reduzir uso de VRAM
    try:
        pipe.enable_attention_slicing()  # reduz uso de memória em attenzione
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    # Se GPU disponível, habilite offload para reduzir VRAM
    if CUDA_AVAILABLE:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            # caso não suportado, ignore
            pass

        # opcional: xformers (se disponível no seu ambiente)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

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
def controlnet_input_ui():
    st.markdown("### Control Type")
    st.markdown("---")
    control_types_ui = ["ALL"] + list(CONTROL_TYPES.keys())

    col1, _, col2 = st.columns([0.35, 0.02, 0.63])
    with col1:
        for ct in control_types_ui:
            is_selected = ct in st.session_state.selected_controls
            # botão com key único
            clicked = st.button(
                ct,
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                key=f"btn_{ct}"
            )
            if clicked:
                # comportamentos
                if ct == "ALL":
                    # selecionar todos ou limpar todos
                    all_set = set(control_types_ui)
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

    for i, style_name in enumerate(['Yarn', 'Rembrandt', 'Van Gogh']):
        render_button(style_name, cols[i % 3], f"style_{i}")

    return st.session_state.selected_style

# ----------------- Preprocess images (lazy detectors) -----------------
def get_preprocessor(name: str):
    """Retorna (e cria se necessario) o preprocessor para 'name' e cacheia em session_state."""
    if name in st.session_state.preprocessors:
        return st.session_state.preprocessors[name]
    cfg = CONTROL_TYPES.get(name)
    if cfg is None:
        return None
    factory = cfg.get("pre_factory")
    if factory is None:
        return None
    # cria e guarda
    proc = factory()
    st.session_state.preprocessors[name] = proc
    return proc

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
        if ctl not in CONTROL_TYPES:
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

        outputs.append(cond_img)

    return outputs
# ----------------- Geração única (pipeline cacheada por lista de controlnets) -----------------
def generate_image(prompt: str, cond_images: List[np.ndarray], selected_controls: List[str], seed: int = 123):
    """
    Gera uma imagem usando pipeline cacheada (dependente do conjunto de controlnets).
    Retorna PIL Image.
    """
    # escolher dtype: se GPU, float16; caso contrário float32
    torch_dtype = torch.float16 if CUDA_AVAILABLE else torch.float32

    # Carrega pipeline (cacheada por control_list)
    pipe = load_pipeline_with_controlnet(CKPT_ID, selected_controls, torch_dtype=torch_dtype)
    print('ok')
    # opcional: tentar carregar LoRA (não obrigatório)
    try_load_lora_weights(pipe, LORA_PATH, LORA_FILENAME)

    # mover para dispositivo (a load_pipeline já tem offload; chamar .to pode forçar toda a pipeline na GPU -> evitar)
    # pipe.to(DEVICE)  # geralmente não necessário se enable_model_cpu_offload foi ativado

    generator = None
    if CUDA_AVAILABLE:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = torch.Generator("cpu").manual_seed(seed)
    print(len(cond_images))
    print(selected_controls)
    # Executa
    result = pipe(
        prompt=prompt,
        image=cond_images if len(cond_images) > 0 else None,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=[1.0] * max(1, len(cond_images)),
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
    gc.collect()

    return out_img

# ----------------- LAYOUT PRINCIPAL -----------------
st.sidebar.markdown("## Configurações")
st.sidebar.write(f"Device: **{DEVICE}**")
if CUDA_AVAILABLE:
    st.sidebar.write("CUDA disponível — usando optimizações (offload, attention slicing quando possível).")
else:
    st.sidebar.write("Sem CUDA — usando CPU (mais lento).")

# ControlNet + input
controlnet_input_ui()

# Styles
_ = styles_ui()

st.markdown("---")
st.markdown("### Prompt")
prompt = st.text_input(label="Prompt", value="A fantasy portrait, cinematic lighting")

# Exibe imagens condicionais pré-processadas (se houver)
if st.session_state.ref_image is not None and len(st.session_state.selected_controls) > 0:
    img = st.session_state.ref_image
    control_images_preview = {}
    for ctrl in st.session_state.selected_controls:
        if ctrl in CONTROL_TYPES:
            proc = get_preprocessor(ctrl)
            if proc is None:
                continue
            processed = proc(img)
            control_images_preview[ctrl] = processed
            st.image(processed, caption=f"ControlNet: {ctrl}", width=240)

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
            final = generate_image(prompt, cond_images, list(st.session_state.selected_controls), seed=123)
            st.image(final, caption="Imagem Final", width=512)
        except Exception as e:
            st.error(f"Erro durante a geração: {e}")
            # tenta liberar memória
            try:
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
