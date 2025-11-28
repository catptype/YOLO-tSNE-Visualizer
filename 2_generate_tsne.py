import os
import glob

from util.YoloFeatureExtractor import YoloFeatureExtractor

# --- CONFIGURATION
MODEL_PATH = r"models\yolo11n-cls.pt"
DATA_DIR = r"DATASET\example"
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

# Paste the layer name you found in yolo_inspect.py here:
TARGET_LAYER = "model.model.10.linear" 

def main():
    # 1. Initialize
    extractor = YoloFeatureExtractor(model_path=MODEL_PATH)
    
    # 2. Gather Images
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    
    all_images = sorted(all_images)
    print(f"Found {len(all_images)} images.")

    if len(all_images) == 0:
        print("No images found. Please check your DATA_DIR.")
        return
    
    # 3. Extract Features (Safely with Caching)
    # Generate a clean ID for the cache file (removes folders from model path)
    model_name = os.path.basename(MODEL_PATH)
    model_id = os.path.splitext(model_name)[0]

    features, paths = extractor.extract_features(
        image_paths=all_images, 
        layer_name=TARGET_LAYER, 
        batch_size=64,       # Adjust if you run out of VRAM
        cache_id=model_id    # Ensures cache uniqueness per model
    )

    # 4. Compute t-SNE
    tsne_xy = extractor.compute_tsne(features, perplexity=30)
    
    # 5. Export Results
    # Create a clean filename: tsne_{model_name}_{layer}.json
    layer_clean = TARGET_LAYER.replace('.', '_')
    output_filename = f"tsne_{model_id}_{layer_clean}"
    
    extractor.export_data(output_filename, paths, tsne_xy)
    
if __name__ == "__main__":
    main()