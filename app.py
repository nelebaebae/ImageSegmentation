import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.title("Image Segmentation dengan MediaPipe")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar asli", use_column_width=True)

    image_np = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    base_options = python.BaseOptions(model_asset_path="deeplab_v3.tflite")
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)

    result = segmenter.segment(mp_image)
    category_mask = result.category_mask

    mask = category_mask.numpy_view()
    condition = np.stack((mask,) * 3, axis=-1) > 0.1

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_rgb, (55, 55), 0)
    output = np.where(condition, image_rgb, blurred)

    st.image(output, caption="Gambar dengan latar belakang blur", use_column_width=True)
