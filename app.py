import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import fitz  # PyMuPDF for PDF processing
from model_loader import load_model  # Assuming the model loader code is saved in model_loader.py

# Function to read PDF and convert it into images
def pdf_to_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        # Convert PyMuPDF pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# Inference function
def run_inference(model, image):
    # Ensure image is in correct format (RGB or BGR)
    results = model.predict(source=image, conf=0.5)
    result_image = results[0].plot()  # Get the resulting image
    return result_image

# Streamlit app
def main():
    st.title("Document Layout Analysis")
    st.write("Upload a PDF or an image to analyze its layout.")

    # Load the model
    model = load_model()

    if model:
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

                # Run inference
                st.write("Running inference...")
                result_image = run_inference(model, image_np)

                # Display results
                st.image(result_image, caption="Detection Results", use_column_width=True)
        else:
            st.error("Please upload a file (PDF or image) to analyze.")
    else:
        st.error("Model could not be loaded.")

if __name__ == "__main__":
    main()
