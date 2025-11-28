import os
from util.YoloFeatureExtractor import YoloFeatureExtractor
from ultralytics import YOLO


# --- CONFIGURATION
MODEL_PATH = r"models\yolo11n-cls.pt"

def main():
    print(f"Inspecting Model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Initialize and Inspect
    extractor = YoloFeatureExtractor(model_path=MODEL_PATH)
    extractor.inspect_layers()
    
    print("\nCopy the 'Layer' name of your choice (e.g., model.model.22) into yolo_tsne.py")

if __name__ == "__main__":
    main()