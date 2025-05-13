# Simple Portal using Streamlit for Tumor Detection and Measurement in Mouse MRI

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Page configuration
st.set_page_config(page_title="Tumor Segmentation Portal", layout="centered")

# Sidebar information
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    This tool allows researchers to upload preclinical mouse MRI images
    and automatically detect tumor regions with basic segmentation.

    Developed for testing proof-of-concept with PDMR-like datasets.
    """)

st.title("🧠 MRI Tumor Analyzer for Preclinical Mouse Models")

st.markdown("""
Upload a mouse MRI image. This tool will automatically detect the tumor region and measure its approximate diameter and area.
""")

# File uploader for MRI image
uploaded_file = st.file_uploader("📤 Upload Mouse MRI Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to grayscale array
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    st.subheader("🔍 Uploaded Image:")
    st.image(image, caption="Mouse MRI", use_column_width=True)
    st.markdown(f"**Image Size:** {img_array.shape[1]} x {img_array.shape[0]} pixels")

    # Apply simple thresholding to highlight bright tumor region
    _, thresh = cv2.threshold(img_array, 120, 255, cv2.THRESH_BINARY)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    if contours:
        # Select the largest contour assuming it's the tumor
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = radius * 2

        # Draw detected tumor contour and enclosing circle on image
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 0, 255), 2)
        cv2.circle(contour_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        st.subheader("📏 Analysis Output:")
        st.image(contour_img, caption="Detected Tumor Region", use_column_width=True)
        st.markdown(f"- **Approximate Diameter:** {diameter:.2f} pixels")
        st.markdown(f"- **Area:** {area:.2f} square pixels")

        # Option to download result
        result_image = Image.fromarray(contour_img)
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        st.download_button(
            label="📥 Download Result Image",
            data=img_buffer.getvalue(),
            file_name="tumor_result.png",
            mime="image/png"
        )
    else:
        st.warning("❗ No tumor region detected. Please try a clearer image.")
