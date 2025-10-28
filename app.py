import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# App Title
st.title("🖼️ Text to Image Generator")
st.write("Generate stunning images from text using Stable Diffusion!")

# Input text prompt
prompt = st.text_area("Enter your image prompt:", "A futuristic city at sunset, cinematic lighting")

# Model choice
model_name = "runwayml/stable-diffusion-v1-5"

@st.cache_resource
def load_model():
    # Load Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("🎨 Generating your image... please wait..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_container_width=True)
            st.success("✅ Image generation complete!")
    else:
        st.warning("⚠️ Please enter a prompt before generating.")
