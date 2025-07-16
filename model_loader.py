import os
import gdown
from ultralytics import YOLO

MODEL_DIR = "models"
FILE_PATH = os.path.join(MODEL_DIR, "yolov10x_best.pt")
FILE_ID = "16dRc24_GxBtSGKfnPyBe0gqsnShouSMD"
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

def download_model():
    
    if not os.path.exists(FILE_PATH):
        print("Downloading model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        try:
            gdown.download(FILE_URL, FILE_PATH, quiet=False, fuzzy=True)
            
            if os.path.exists(FILE_PATH) and os.path.getsize(FILE_PATH) > 0:
                print(f"Model downloaded successfully at: {FILE_PATH}")
            else:
                print("Download failed or file is corrupt.")
                return None
        except Exception as e:
            print(f"Error downloading the model: {e}")
            return None
    else:
        print(f"Model already exists at: {FILE_PATH}")

    return FILE_PATH

def load_model():
    model_path = download_model()
    
    if model_path:
        try:
            model = YOLO(model_path, task="detect")  
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("Model download failed, unable to load.")
        return None

if __name__ == "__main__":
    model = load_model()
    if model:
        print("Model is ready for inference.")
    else:
        print("Failed to load the model.")
