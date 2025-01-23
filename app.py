import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import fitz  # PyMuPDF for PDF processing

weights_path = "https://github.com/AnkitHarsh54/Document-Segmentation/blob/main/yolov10x_best.pt"
# Load the YOLO model
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

# Function to read PDF and convert it into images
def pdf_to_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# Inference function
def run_inference(model, image):
    results = model.predict(source=image, conf=0.5)
    result_image = results[0].plot()
    return result_image

# Streamlit app
def main():
    st.title("Document Layout Analysis")
    st.write("Upload a PDF or an image to analyze its layout.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file (PDF or image)...", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "pdf":
            # Process PDF
            st.write("Converting PDF to images...")
            images = pdf_to_images(uploaded_file)
            st.write(f"Extracted {len(images)} page(s).")

            # Display and process each page
            for page_num, img in enumerate(images, start=1):
                st.write(f"Page {page_num}:")
                st.image(img, caption=f"Page {page_num}", use_column_width=True)

                # Convert PIL image to NumPy array for YOLO
                image_np = np.array(img)

                # Load the model
                model = load_model(weights_path)

                # Run inference
                st.write("Running inference...")
                result_image = run_inference(model, image_np)

                # Display results
                st.image(result_image, caption=f"Detection Results - Page {page_num}", use_column_width=True)
        else:
            # Process image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert PIL image to NumPy array for YOLO
            image_np = np.array(image)

            # Load the model
            model = load_model("")

            # Run inference
            st.write("Running inference...")
            result_image = run_inference(model, image_np)

            # Display results
            st.image(result_image, caption="Detection Results", use_column_width=True)

if __name__ == "__main__":
    main()
