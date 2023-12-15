import streamlit as st 
from PIL import Image 
from io import BytesIO
import base64
import numpy as np

import backend

st.title('Trace')

style_names = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
style_name = st.sidebar.selectbox('Select Style', options=style_names)

model = "pretrained_models/" + style_name + ".pth" 

uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

cols = st.columns((1,1))
with cols[0]: 
    input_image = st.empty() 
with cols[1]: 
    transformed_image = st.empty()

if uploaded_image is not None: 
    clicked = st.sidebar.button("Trace")
    pil_image = Image.open(uploaded_image) 
    image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

    input_image.image(uploaded_image)

    if clicked: 
        model = backend.load_models(model)
        result = backend.transform(model, input=uploaded_image)

        st.write("### Output image:")
        st.image(result)
        
        