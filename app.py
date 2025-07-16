import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import fitz  
from model_loader import load_model  
import tempfile
import cv2

def pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())  
        temp_pdf_path = temp_pdf.name

    doc = fitz.open(temp_pdf_path)  
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL Image
        images.append(img)

    return images

def run_inference(model, image):
    results = model.predict(source=image, conf=0.5)
    result_image = results[0].plot()  
    
    result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    return result_pil

def main():
    st.title("📄 Document Layout Analysis")
    st.write("Upload a PDF or an image to analyze its layout using YOLO.")

    model = load_model()
    if not model:
        st.error("⚠️ Model could not be loaded. Check the logs for details.")
        return

    uploaded_file = st.file_uploader("📂 Choose a file (PDF or image)...", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type == "pdf":
            
            st.write("📜 Converting PDF to images...")
            images = pdf_to_images(uploaded_file)
            st.write(f"📄 Extracted {len(images)} page(s).")

            for page_num, img in enumerate(images, start=1):
                st.write(f"🖼️ Page {page_num}:")
                st.image(img, caption=f"Page {page_num}", use_column_width=True)

                image_np = np.array(img)
                
                st.write("🔍 Running inference...")
                result_image = run_inference(model, image_np)

                st.image(result_image, caption=f"📝 Detection Results - Page {page_num}", use_column_width=True)

        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

            image_np = np.array(image)

            st.write("🔍 Running inference...")
            result_image = run_inference(model, image_np)

            st.image(result_image, caption="📝 Detection Results", use_column_width=True)
    else:
        st.warning("⚠️ Please upload a file (PDF or image) to analyze.")

if __name__ == "__main__":
    main()
