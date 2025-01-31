import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import fitz  # PyMuPDF for PDF processing
from model_loader import load_model  
import tempfile
import cv2

# Function to read PDF and convert it into images
def pdf_to_images(pdf_file):
    """Convert PDF pages to images."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())  # Save uploaded PDF to temp file
        temp_pdf_path = temp_pdf.name

    doc = fitz.open(temp_pdf_path)  # Open saved file
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image
        images.append(img)

    return images

# Inference function
def run_inference(model, image):
    """Run YOLO inference and return annotated image."""
    results = model.predict(source=image, conf=0.5)
    result_image = results[0].plot()  # Get result as NumPy array

    # Convert to PIL Image for Streamlit display
    result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    return result_pil

# Streamlit app
def main():
    st.title("📄 Document Layout Analysis")
    st.write("Upload a PDF or an image to analyze its layout using YOLO.")

    # Load YOLO model
    model = load_model()
    if not model:
        st.error("⚠️ Model could not be loaded. Check the logs for details.sdbdsb")
        return

    # File uploader
    uploaded_file = st.file_uploader("📂 Choose a file (PDF or image)...", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "pdf":
            # Process PDF
            st.write("📜 Converting PDF to images...")
            images = pdf_to_images(uploaded_file)
            st.write(f"📄 Extracted {len(images)} page(s).")

            # Display and process each page
            for page_num, img in enumerate(images, start=1):
                st.write(f"🖼️ Page {page_num}:")
                st.image(img, caption=f"Page {page_num}", use_column_width=True)

                # Convert to NumPy for YOLO
                image_np = np.array(img)

                # Run inference
                st.write("🔍 Running inference...")
                result_image = run_inference(model, image_np)

                # Display results
                st.image(result_image, caption=f"📝 Detection Results - Page {page_num}", use_column_width=True)

        else:
            # Process single image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

            # Convert to NumPy for YOLO
            image_np = np.array(image)

            # Run inference
            st.write("🔍 Running inference...")
            result_image = run_inference(model, image_np)

            # Display results
            st.image(result_image, caption="📝 Detection Results", use_column_width=True)
    else:
        st.warning("⚠️ Please upload a file (PDF or image) to analyze.")

if __name__ == "__main__":
    main()
