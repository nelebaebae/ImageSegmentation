import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import os

st.title("Segmentasi Gambar: Upload atau Kamera")

# Load model
MODEL_PATH = "deeplab_v3.tflite"
if not os.path.exists(MODEL_PATH):
    st.error("Model TFLite tidak ditemukan.")
    st.stop()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Pilihan input
input_mode = st.radio("Pilih sumber gambar:", ["Upload Gambar", "Gunakan Kamera"])

def segment_image(image_np):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view()
    condition = np.stack((mask,) * 3, axis=-1) > 0.1
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_bgr, (55, 55), 0)
    output = np.where(condition, image_bgr, blurred)
    return output

# Upload
if input_mode == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar asli", use_column_width=True)
        image_np = np.array(image)
        result_image = segment_image(image_np)
        st.image(result_image, caption="Hasil Segmentasi", channels="BGR", use_column_width=True)

# Kamera
else:
    start_cam = st.button("Aktifkan Kamera")
    if start_cam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame = segment_image(frame_rgb)
            stframe.image(result_frame, channels="BGR", use_column_width=True)

            # Tombol keluar
            if st.button("Stop Kamera"):
                break

        cap.release()
        cv2.destroyAllWindows()
