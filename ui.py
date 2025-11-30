import streamlit as st
import numpy as np
import cv2
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import (
    CannyDetector, OpenposeDetector, MidasDetector,
    MLSDdetector, LineartDetector, HEDdetector
)

from streamlit_drawable_canvas import st_canvas

# ======================= CONFIG ===========================
st.set_page_config(page_title="ControlNet UI", layout="wide")
st.title("ControlNet – Upload + Desenho")

st.markdown("### Control Type")

# Tipos disponíveis na UI
control_types = ["Canny", "Depth", "OpenPose", "MLSD", "Lineart", "SoftEdge"]

# ======================= MODELOS ===========================

CONTROLNET_MODELS = {
    "Canny": "lllyasviel/sd-controlnet-canny",
    "Depth": "lllyasviel/sd-controlnet-depth",
    "OpenPose": "lllyasviel/sd-controlnet-openpose",
    "MLSD": "lllyasviel/sd-controlnet-mlsd",
    "Lineart": "lllyasviel/sd-controlnet-lineart",
    "SoftEdge": "lllyasviel/sd-controlnet-hed",
}

PREPROCESSORS = {
    "Canny": CannyDetector(),
    # "Depth": MidasDetector("lllyasviel/midas"),
    # "OpenPose": OpenposeDetector(),
    # "MLSD": MLSDdetector(),
    # "Lineart": LineartDetector(),
    # "SoftEdge": HEDdetector(),
}

# ======================= ESTADO ===========================

if "selected" not in st.session_state:
    st.session_state.selected = None
if "ref_image" not in st.session_state:
    st.session_state.ref_image = None


# ==================== BOTÕES CONTROLNET ====================

cols = st.columns(6)

def render_button(label, col):
    clicked = col.button(
        label,
        use_container_width=True,
        type="primary" if st.session_state.selected == label else "secondary",
    )
    if clicked:
        st.session_state.selected = label

col_i = 0
for ct in control_types:
    render_button(ct, cols[col_i])
    col_i += 1
    if col_i == len(cols):
        cols = st.columns(6)
        col_i = 0

selected = st.session_state.selected

# ======================= INPUT MODE ===========================
st.markdown("---")
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

# ======================= PROMPT ===========================
st.markdown("---")
st.markdown("### Prompt")

user_prompt = st.text_input(label='Prompt')
style = st.selectbox('Styles', ['rembrandt', 'vangogh'])


# # ======================= PRÉ-PROCESSAMENTO ===================

# if st.session_state.ref_image is not None and selected is not None:

#     st.markdown("---")
#     st.markdown(f"### 3. Pré-processamento: **{selected}**")

#     image = st.session_state.ref_image
#     preprocessor = PREPROCESSORS[selected]
#     cond_image = preprocessor(image)

#     st.image(cond_image, caption="Imagem pré-processada", width=400)


# # ======================= GERAÇÃO ============================

# if st.session_state.ref_image is not None and selected is not None:
#     st.markdown("---")
#     st.markdown("### 4. Geração")

#     prompt = st.text_input("Prompt", "a beautiful portrait, detailed, 4k")

#     if st.button("Gerar imagem"):
#         model_name = CONTROLNET_MODELS[selected]

#         with st.spinner("Carregando modelos..."):
#             controlnet = ControlNetModel.from_pretrained(
#                 model_name, torch_dtype=torch.float16
#             )

#             pipe = StableDiffusionControlNetPipeline.from_pretrained(
#                 "runwayml/stable-diffusion-v1-5",
#                 controlnet=controlnet,
#                 torch_dtype=torch.float16,
#             ).to("cuda")

#         with st.spinner("Gerando..."):
#             out = pipe(
#                 prompt=prompt,
#                 image=cond_image,
#                 num_inference_steps=30,
#                 controlnet_conditioning_scale=1.0,
#             ).images[0]

#         st.image(out, caption="Resultado Final", width=450)

# =


