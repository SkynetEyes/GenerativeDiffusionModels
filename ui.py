from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers import BitsAndBytesConfig
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline
import streamlit as st
import torch
import gc
import numpy as np
from controlnet_aux import (
    CannyDetector, OpenposeDetector,  ZoeDetector, LineartDetector
)
from streamlit_drawable_canvas import st_canvas
import cv2

CKPT_ID = "CompVis/stable-diffusion-v1-4"
LORA_PATH = "results"
LORA_FILENAME = "pytorch_lora_weights.safetensors"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ======================= MODELOS ===========================

CONTROL_TYPES = {
    "Canny": {
        "model": "lllyasviel/sd-controlnet-canny",
        "pre": CannyDetector(),
    },
    "Depth": {
        "model": "lllyasviel/control_v11f1p_sd15_depth" ,
        "pre": ZoeDetector.from_pretrained("lllyasviel/Annotators"),
    },
    "Pose": {
        "model": "xinsir/controlnet-openpose-sdxl-1.0" ,
        "pre": OpenposeDetector.from_pretrained('lllyasviel/ControlNet'),
    },
    "Segmentation": {
        "model": "lllyasviel/control_v11p_sd15_lineart",
        "pre": LineartDetector.from_pretrained("lllyasviel/Annotators"),
    },
}

@st.cache_resource
def load_base_pipeline():

    # Carregar UNet quantizado
    unet = UNet2DConditionModel.from_pretrained(
        CKPT_ID, subfolder="unet",
        quantization_config=nf4_config,
        torch_dtype=torch.float16
    )

    # Carregar Text Encoder quantizado
    text_encoder = CLIPTextModel.from_pretrained(
        CKPT_ID, subfolder="text_encoder",
        quantization_config=nf4_config,
        torch_dtype=torch.float16
    )

    # Pipeline base sem ControlNet ainda
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        CKPT_ID,
        unet=unet,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipe.enable_attention_slicing("auto")
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    return pipe

def load_controlnets(control_list):
    controlnets = []
    for ctl in control_list:
        model_id = CONTROL_TYPES[ctl]['model']
        model = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        controlnets.append(model)
    return controlnets

def toggle_control(label):
    if label in st.session_state.selected_controls:
        st.session_state.selected_controls.remove(label)
    else:
        st.session_state.selected_controls.add(label)

def controlnet_input():
    st.markdown("### Control Type")
    st.markdown("---")
    # Tipos disponíveis na UI
    control_types = ["ALL","Canny", "Depth", "Pose", "LinearArt"]

    # ==================== BOTÕES CONTROLNET ====================
    col1,_,col2 = st.columns([0.3,0.05,0.65])

    with col1:

        for ct in control_types:
            is_selected = ct in st.session_state.selected_controls

            clicked = st.button(
                ct,
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                key=f"btn_{ct}"
            )

            # Toggle quando clica
            if clicked:
                if ct == "ALL":
                    if len(st.session_state.selected_controls) == len(control_types) :
                        st.session_state.selected_controls = set()
                    else:
                        st.session_state.selected_controls = set(control_types)
                    st.rerun()
                else:
                    if ct in st.session_state.selected_controls:
                        st.session_state.selected_controls.remove(ct)
                    else:
                        st.session_state.selected_controls.add(ct)
                    st.rerun()

    # ======================= INPUT MODE ===========================
    with col2:
        st.markdown("### Selecione o método de entrada")

        input_mode = st.radio("Como deseja fornecer a imagem-base?", ["Upload", "Desenhar"], horizontal=True)

        img = None


        # ======================= UPLOAD ==============================
        if input_mode == "Upload":
            uploaded = st.file_uploader(label='Imagem de Controle',type=["jpg", "jpeg", "png"])

            if uploaded:
                arr = np.frombuffer(uploaded.read(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR→RGB
                st.image(img, caption="Imagem enviada", width=400)
                st.session_state.ref_image = img


        # ======================= DESENHO =============================

        if input_mode == "Desenhar":

            st.markdown("Desenhe sua imagem abaixo:")
            c1,c2 = st.columns(2)

            with c1:
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
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    st.image(img, caption="Desenho capturado", width=400)
                    st.session_state.ref_image = img

def styles():
    cols = st.columns(3)

    def render_button(label, col):
        clicked = col.button(
            label,
            use_container_width=True,
            type="primary" if st.session_state.selected == label else "secondary"
        )
        if clicked:
            st.session_state.selected = label
            st.rerun()

    c = 0
    for ct in ['Yarn','Rembrandt', 'Van Gogh']:
        render_button(ct, cols[c])
        c += 1
        if c == len(cols):
            cols = st.columns(6)
            c = 0

    selected = st.session_state.selected
    return selected

def preprocess_images(ref_image, selected_controls):
    outputs = []
    for ctl in selected_controls:
        if ctl in CONTROL_TYPES:
            proc = CONTROL_TYPES[ctl]['pre']
            cond = proc(ref_image)
            outputs.append(cond)
    return outputs

def generate_image(prompt, control_images, selected_controls, seed=123):

    pipe = load_base_pipeline()
    controlnets = load_controlnets(selected_controls)
    pipe.controlnet = controlnets
    pipe.load_lora_weights(LORA_PATH, weight_name=LORA_FILENAME)
    generator = torch.Generator("cuda").manual_seed(seed)

    out = pipe(
        prompt=prompt,
        image=control_images,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=[1.0] * len(controlnets),
        generator=generator
    ).images[0]

    return out


# ======================= CONFIG ===========================
st.set_page_config(page_title="Diffusion ControlNet UI", layout="wide")
st.title("Arte Controlada e Estilo")

# ======================= ESTADO ===========================

if "selected_controls" not in st.session_state:
    st.session_state.selected_controls = set()
if "selected" not in st.session_state:
    st.session_state.selected = None
if "ref_image" not in st.session_state:
    st.session_state.ref_image = None

# ======================= CONTROLNET ===========================
controlnet_input()

# ======================= Styles ===========================
st.markdown("---")
st.markdown("### Styles")
styles()

# ======================= PROMPT ===========================
st.markdown("---")
st.markdown("### Prompt")

prompt = st.text_input(label='Prompt')

if st.session_state.ref_image is not None and st.session_state.selected_controls:
        img = st.session_state.ref_image
        control_images = {}

        for ctrl in st.session_state.selected_controls:
            if ctrl in CONTROL_TYPES:
                processed = CONTROL_TYPES[ctrl]['pre'](img)
                control_images[ctrl] = processed
                st.image(processed, caption=f"ControlNet: {ctrl}", width=300)


# ======================= GERAÇÃO ===============================

if st.session_state.ref_image is not None and st.session_state.selected is not None and len(st.session_state.selected_controls) > 0:

    if st.button("Gerar imagem", type="primary"):

        # ---- Carregar todos os modelos selecionados ----
        with st.spinner("Carregando modelos ControlNet..."):

            controlnet_models = []
            for ctrl in st.session_state.selected_controls:
                model_name = CONTROL_TYPES[ctrl]
                model = ControlNetModel.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16
                )
                controlnet_models.append(model)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet_models,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to("cuda")

        # ---- Preparar as imagens pré-processadas ----
        cond_images = [control_images[c] for c in st.session_state.selected_controls]

        # ---- Gerar ----
        with st.spinner("Gerando imagem..."):
            output = pipe(
                prompt=prompt,
                image=cond_images,
                num_inference_steps=30,
                controlnet_conditioning_scale=[1.0] * len(cond_images)
            ).images[0]

        st.image(output, caption="Resultado Final", width=512)

if st.button("Gerar imagem"):

    # 1. Pré-processamento
    control_images = preprocess_images(
        st.session_state.ref_image,
        st.session_state.selected_controls
    )

    # 2. Geração final
    final = generate_image(
        prompt,
        control_images,
        st.session_state.selected_controls
    )

    st.image(final, caption="Imagem Final", width=512)