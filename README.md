# 🕵️ YOLO Dataset Auditor & t-SNE Visualizer

A powerful toolkit to visualize dataset clusters and automatically detect labeling errors using **Ultralytics YOLO** models (v8, v11, v12).

This tool extracts deep feature vectors from your images, projects them into 2D space using t-SNE, and uses **k-Nearest Neighbors (k-NN)** to identify "Suspicious" data points (e.g., a "Cat" image sitting deep inside a "Dog" cluster).

---

## 🌟 Key Features

*   **Universal Support:** Works with **YOLOv8, YOLOv11, YOLOv12** (Detect, Classify, Pose, Segment, OBB).
*   **Smart Inspection:** Automatically identifies the best layers to hook for feature extraction.
*   **Robust Caching:** Supports massive datasets (100k+ images). If interrupted, it **resumes exactly where it left off**.
*   **Ghost Mode Visualization:** A specialized plotting mode that makes clean data transparent and highlights potential errors.
*   **Actionable Reports:** Generates a CSV list of mislabeled images to fix.

---

## 📂 Project Structure

```text
.
├── util/
│   └── YoloFeatureExtractor.py   # Core logic engine
│
├── 1_inspect_model.py            # Step 1: Find the right layer
├── 2_generate_tsne.py            # Step 2: Extract features & t-SNE
├── 3_view_plot.py                # Step 3: Interactive Scatter Plot
├── 4_analyze_errors.py           # Step 4: AI Conflict Detection
│
├── models/
│   └── yolo11n-cls.pt            # (Default) Small classification model
│
├── DATASET/
│   └── example/                  # Included demo dataset
│       ├── cheeseburger/
│       ├── flamingo/
│       ├── violin/
│       └── ... (12 classes total)
│
└── requirements.txt
```

---

## 🚀 Installation

1.  **Clone the repository** (or download the files).
2.  **Install dependencies:**

```bash
pip install ultralytics scikit-learn pandas plotly tqdm
```

*(Note: GPU is recommended for Step 2, but CPU works fine for smaller datasets.)*

---

## 📖 Usage Workflow

This guide uses the included **`DATASET/example`** and **`yolo11n-cls.pt`** so you can run it immediately.

### Step 1: Inspect Your Model
Different YOLO versions have different architectures. Run this to find the best feature layer.

1.  Open `1_inspect_model.py` and ensure `MODEL_PATH = "models/yolo11n-cls.pt"`.
2.  Run the script:
    ```bash
    python 1_inspect_model.py
    ```
3.  **Result:** It identifies the classification vector.
    ```text
    ==================== RECOMMENDED LAYERS ====================
    Layer: model.model.9         | Type: C2PSA      <--- ATTENTION BLOCK
    Layer: model.model.10.linear | Type: Linear     <--- CLASSIFICATION VECTOR
    ============================================================
    ```
    *(Copy `model.model.10.linear` for the next step).*

### Step 2: Generate Data (The Heavy Lifting)
This extracts features from all images and calculates t-SNE coordinates.

1.  Open `2_generate_tsne.py` and configure it:
    ```python
    MODEL_PATH   = "models/yolo11n-cls.pt"
    DATA_DIR     = "./DATASET/example"      # Using the demo data
    TARGET_LAYER = "model.model.10.linear"  # Paste layer from Step 1
    ```
2.  Run:
    ```bash
    python 2_generate_tsne.py
    ```
3.  **Result:**
    ```text
    Found 96 images.
    Extracting features... 100%|████████| 3/3 [00:01<00:00, 2.15it/s]
    Computing t-SNE on (96, 1280)...
    Export Complete: tsne_results/tsne_yolo11n-cls_model_model_10_linear.txt
    ```

### Step 3: View Clusters
Check the general health of your dataset.

1.  Open `3_view_plot.py` and set `JSON_FILE` to the result from Step 2.
2.  Run:
    ```bash
    python 3_view_plot.py
    ```
3.  **Result:** Opens an interactive HTML plot.
    
    ![Cluster Plot](docs/images/screenshot1.png)
    *(Screenshot: Distinct clusters for example dataset.)*

### Step 4: Analyze Errors (Ghost Mode)
Find the wrong labels. This uses k-NN to find images surrounded by the wrong class.

1.  Open `4_analyze_errors.py` and set the `JSON_FILE`.
2.  Run:
    ```bash
    python 4_analyze_errors.py
    ```
3.  **Result:**
    *   **Visual Map (`_focus_map.html`):**
        ![Ghost Mode](docs/images/screenshot2.png)
        *(Screenshot: Suspicious points appear as Red Diamonds, clean data is faint grey)*
    *   **Fix List (`_fix_list.csv`):**
        
        | filename | class | suggestion | conflict_score |
        | :--- | :--- | :--- | :--- |
        | `hotdog_006.jpg` | `hotdog` | `Likely: cheeseburger` | `0.8` |
        | `corn_001.jpg` | `corn` | `Likely: hotdog` | `1.0` |

---

## 🛠️ Configuration & Tips

*   **⚠️ Dataset Size Requirement (CRITICAL):**
    *   t-SNE **will fail** if your dataset is smaller than the Perplexity value.
    *   **Requirement:** You must have **more images than the Perplexity setting** (default 30).
    *   **Recommendation:** For meaningful clusters, use a dataset with **at least 100 images**. If you have fewer, the visualization will likely be poor or the script will crash.
*   **Handling Large Datasets (100k+ images):**
    *   Step 2 is safe to interrupt. It caches progress in the `cache/` folder.
    *   If you *retrain* your model, **rename the model file** (e.g., `v1.pt` -> `v2.pt`) to force the script to ignore the old cache.
*   **Perplexity:**
    *   In `2_generate_tsne.py`, set `PERPLEXITY = 30` (default).
    *   For huge datasets, you can try `50`.

---

## License

MIT License. Feel free to modify and use for your own projects!