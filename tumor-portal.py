# Tumor Segmentation Portal with DICOM Conversion and Improved Detection

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import pydicom
import os

# Page configuration
st.set_page_config(page_title="Tumor Segmentation Portal", layout="centered")

# Sidebar information
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    This tool allows researchers to upload preclinical mouse MRI images
    and automatically detect tumor regions.

    ✅ Supports JPEG, PNG, and DICOM formats.
    ✅ Uses enhanced contour filtering to reduce false positives.
    
    ---
    🧬 **Developed by Intellogene Group**  
    📧 Contact: intellogene@gmail.com
    """)

st.title("🧠 MRI Tumor Analyzer for Preclinical Mouse Models")

st.markdown("""
Upload a mouse MRI image (JPEG/PNG) or a DICOM file (.dcm). This tool detects tumor regions and estimates their diameter and area.
""")

# File uploader for image or DICOM
uploaded_file = st.file_uploader("📤 Upload Mouse MRI (JPEG, PNG, or DICOM)", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_ext == ".dcm":
        # Load and convert DICOM to image
        ds = pydicom.dcmread(uploaded_file)
        image_array = ds.pixel_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        img_array = np.array(image)
        st.info("DICOM file converted to image.")
    else:
        # Convert uploaded image to grayscale array
        image = Image.open(uploaded_file).convert("L")
        img_array = np.array(image)

    st.subheader("🔍 Uploaded Image:")
    st.image(img_array, caption="Mouse MRI", use_column_width=True)
    st.markdown(f"**Image Size:** {img_array.shape[1]} x {img_array.shape[0]} pixels")

    # Apply adaptive threshold and contour filtering
    blur = cv2.GaussianBlur(img_array, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # Filter contours by area to reduce false detection
    filtered = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 10000]

    if filtered:
        largest_contour = max(filtered, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = radius * 2

        cv2.drawContours(contour_img, [largest_contour], -1, (0, 0, 255), 2)
        cv2.circle(contour_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        st.subheader("📏 Analysis Output:")
        st.image(contour_img, caption="Detected Tumor Region", use_column_width=True)
        st.markdown(f"- **Approximate Diameter:** {diameter:.2f} pixels")
        st.markdown(f"- **Area:** {area:.2f} square pixels")

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
        st.warning("❗ No tumor-like region detected. Try another image or adjust filtering.")

