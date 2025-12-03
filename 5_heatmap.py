import os
import glob
from util.YoloFeatureExtractor import YoloFeatureExtractor

# --- CONFIGURATION
MODEL_PATH = r"models\yolo11n-cls.pt"
DATA_DIR = r"DATASET\example"
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

def main():
    # 1. Setup
    # Ensure you have 'opencv-python' installed: pip install opencv-python
    
    # 2. Gather Images
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    
    all_images = sorted(all_images)
    print(f"Found {len(all_images)} images.")

    test_image = all_images[0] # Pick the first image found
    print(f"Processing image: {test_image}")

    # 2. Load Model
    extractor = YoloFeatureExtractor(model_path=MODEL_PATH)
    
    # 3. Define Layers to Visualize
    # Use extractor.inspect_layers() to find others.
    layers_to_check = [
        "model.model.0",   # Very first convolution (Raw edges)
        "model.model.2",   # C3k2
        "model.model.4",   # C3k2
        "model.model.6",   # C3k2
        "model.model.8",   # C3k2
        "model.model.9",   # ????
    ]

    # 4. Generate & Save
    output_folder = "heatmap_results"
    print(f"Saving results to '{output_folder}'...")
    
    for layer in layers_to_check:
        print(f"--> Generating heatmap for {layer}...")
        extractor.generate_heatmap(
            image_path=test_image, 
            layer_name=layer, 
            output_dir=output_folder
        )

    print("\nDone! Open the folder to see the images.")

if __name__ == "__main__":
    main()