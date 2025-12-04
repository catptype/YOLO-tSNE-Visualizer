import cv2
import os
import torch
import json
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.manifold import TSNE

class YoloFeatureExtractor:
    def __init__(self, model_path: str, task: str = None):
        self.model = YOLO(model_path, task=task)
        self.device = self.model.device
        self._hook_handle = None
        self._batch_features = []
        print(f"Model loaded: {model_path} on {self.device}")

    def inspect_layers(self):
        """
        Smartly prints available layers.
        """
        print(f"{'='*20} RECOMMENDED LAYERS {'='*20}")
        torch_ignore = ['Conv2d', 'BatchNorm2d', 'SiLU', 'Identity', 'ModuleList', 'Sequential', 'Upsample', 'Concat', 'DFL', 'MaxPool2d', 'Dropout', 'ConvTranspose2d']
        ultralytics_ignore = ['Conv', 'DWConv', 'Bottleneck', 'C3k', 'RepC3', 'GhostConv', 'Attention', 'PSABlock', 'TransformArgs', 'Proto', 'ABlock', 'AAttn']
        ignore_types = torch_ignore + ultralytics_ignore
        
        for name, module in self.model.named_modules():
            mod_type = module.__class__.__name__
            if name == "" or name == "model": continue
            if mod_type in ignore_types: continue
                
            suffix = ""
            if mod_type == 'Linear': suffix = " <--- CLASSIFICATION VECTOR"
            elif mod_type == 'AdaptiveAvgPool2d': suffix = " <--- PRE-VECTOR POOLING"
            elif mod_type in ["Detect", "Segment", "Pose", "OBB", "Classify"]: suffix = " <--- HEAD"
            elif "SPPF" in mod_type: suffix = " <--- BACKBONE END"
            elif mod_type == "C2PSA": suffix = " <--- ATTENTION BLOCK"
            elif name.endswith(".20") or name.endswith(".21") or name.endswith(".22"): suffix = " <--- NECK END"

            print(f"Layer: {name:<25} | Type: {mod_type:<15}{suffix}")
        print(f"{'='*60}")

    def _feature_hook(self, module, input, output):
        if isinstance(output, tuple): data = output[0]
        else: data = output
        
        if isinstance(data, list): data = data[-1] 

        if data.dim() == 4: data = torch.mean(data, dim=[2, 3])
        elif data.dim() == 3: data = torch.mean(data, dim=1)

        self._batch_features.append(data.detach().cpu())

    def _load_cache(self, cache_path: str) -> Dict[str, torch.Tensor]:
        """Loads the checkpoint dictionary if it exists."""
        if os.path.exists(cache_path):
            print(f"Resuming from cache: {cache_path}")
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(f"Warning: Cache file corrupted ({e}). Starting fresh.")
                return {}
        return {}

    def _save_cache(self, cache_data: Dict[str, torch.Tensor], cache_path: str):
        """
        Atomic save: Write to temp file then rename. 
        Prevents corruption if script crashes during write.
        """
        tmp_path = cache_path + ".tmp"
        torch.save(cache_data, tmp_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        os.rename(tmp_path, cache_path)

    def extract_features(self, image_paths: List[str], layer_name: str, batch_size: int = 32, cache_id: str = "default") -> Tuple[np.ndarray, List[str]]:
        """
        Extracts features with safety checkpoints.
        """
        # 1. Setup Cache
        os.makedirs("cache", exist_ok=True)
        # Create a unique filename based on model and layer so we don't mix up features
        cache_filename = f"cache/feat_{cache_id}_{layer_name.replace('.', '_')}.pt"
        feature_cache = self._load_cache(cache_filename)
        
        # 2. Filter images that are already processed
        images_to_process = [p for p in image_paths if p not in feature_cache]
        print(f"Total Images: {len(image_paths)}")
        print(f"Cached:       {len(feature_cache)}")
        print(f"To Process:   {len(images_to_process)}")

        if len(images_to_process) > 0:
            # 3. Register Hook
            found = False
            for name, module in self.model.named_modules():
                if name == layer_name:
                    self._hook_handle = module.register_forward_hook(self._feature_hook)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Layer '{layer_name}' not found.")

            self._batch_features = []
            
            # 4. Process in Batches
            total_batches = (len(images_to_process) + batch_size - 1) // batch_size
            save_interval = 10 # Save to disk every 10 batches (adjust for speed vs safety)
            
            try:
                for i in tqdm(range(0, len(images_to_process), batch_size), total=total_batches, desc="Extracting"):
                    batch_paths = images_to_process[i : i + batch_size]
                    
                    # Inference
                    self.model.predict(
                        source=batch_paths, 
                        verbose=False, 
                        device=self.device,
                        stream=False,
                        batch=len(batch_paths)
                    )
                    
                    # Map results to paths
                    current_features = torch.cat(self._batch_features, dim=0) # tensor (Batch, Dim)
                    
                    # Update Memory Cache
                    for path, feat in zip(batch_paths, current_features):
                        feature_cache[path] = feat
                    
                    # Clear buffer
                    self._batch_features = []
                    
                    # Periodic Disk Save
                    if (i // batch_size) % save_interval == 0:
                        self._save_cache(feature_cache, cache_filename)

            except KeyboardInterrupt:
                print("\n\n!!! INTERRUPTED !!! Saving progress before exiting...")
                self._save_cache(feature_cache, cache_filename)
                print("Progress saved. Run again to resume.")
                exit(0)
            except Exception as e:
                print(f"Error: {e}")
                self._save_cache(feature_cache, cache_filename)
                raise e
            finally:
                if self._hook_handle:
                    self._hook_handle.remove()
                # Final save at the end of loop
                self._save_cache(feature_cache, cache_filename)

        # 5. Assemble Final Arrays in the correct order of 'image_paths'
        # This ensures the output matches the input list order, even if processed out of order
        final_features = []
        valid_paths = []
        
        print("Assembling final feature matrix...")
        for path in image_paths:
            if path in feature_cache:
                final_features.append(feature_cache[path])
                valid_paths.append(path)
        
        if not final_features:
            raise RuntimeError("No features available.")

        # Stack into numpy array
        return torch.stack(final_features).numpy(), valid_paths

    def compute_tsne(self, features: np.ndarray, perplexity: int = 30, xyz: bool = False) -> np.ndarray:
        n_samples = features.shape[0]
        if n_samples <= perplexity:
            perplexity = max(1, n_samples - 1)
        
        n_components = 3 if xyz else 2
        print(f"Computing t-SNE on {features.shape}... in {n_components}D")
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
        return tsne.fit_transform(features)

    def export_data(self, output_name: str, paths: List[str], tsne_results: np.ndarray):
        output_dir = "tsne_results"
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, f"{output_name}.json")
        txt_path = os.path.join(output_dir, f"{output_name}.txt")

        # Check dimensions (2 or 3)
        dims = tsne_results.shape[1]

        # Define dynamic axis names
        if dims == 2:
            axis_keys = ['x', 'y']
        elif dims == 3:
            axis_keys = ['x', 'y', 'z']
        else:
            # Fallback for 4+ dimensions: d0, d1, d2, ...
            axis_keys = [f"d{i}" for i in range(dims)]
        
        data_list = []
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            # 2. Write Header dynamically
            # Example: x|y|class|filename... OR x|y|z|class|filename...
            header = "|".join(axis_keys + ["class", "filename", "fullpath"])
            f_txt.write(header + "\n")
            for path, point in zip(paths, tsne_results):
                class_name = os.path.basename(os.path.dirname(path))
                filename = os.path.basename(path)
                
                # 3. Build Dictionary
                entry = {}
                coord_strings = []
                
                # Add coordinates dynamically
                for i, key in enumerate(axis_keys):
                    val = float(point[i])
                    entry[key] = val
                    coord_strings.append(f"{val:.5f}")
                
                # Add metadata
                entry["class"] = class_name
                entry["filename"] = filename
                entry["path"] = path
                
                data_list.append(entry)
                
                # 4. Write Line to Text File
                # Join coordinates + metadata with pipe
                line_data = coord_strings + [class_name, filename, path]
                f_txt.write("|".join(line_data) + "\n")
                
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(data_list, f_json, indent=4)
        print(f"Export Complete ({dims}D): {txt_path}")
    
    def generate_heatmap(self, image_path: str, layer_name: str, output_dir: str = "heatmap_output"):
        """
        Generates a heatmap and SAVES it to disk (No plt.show).
        """
        
        
        # 1. Setup Hook
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple): data = output[0]
                elif isinstance(output, list): data = output[-1]
                else: data = output
                activation[name] = data.detach()
            return hook

        handle = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(get_activation(layer_name))
                break
        
        if handle is None:
            print(f"Layer {layer_name} not found.")
            return

        # 2. Prepare Image (The "Squash" Step)
        # We load the image using OpenCV to get exact dimensions
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            print(f"Error: Cannot read {image_path}")
            return
            
        orig_h, orig_w = img_raw.shape[:2]

        # We manually resize to 640x640. 
        # This distorts the image slightly, but eliminates the black padding bars 
        # that were causing your alignment issues.
        input_size = (640, 640)
        img_resized = cv2.resize(img_raw, input_size)

        # 3. Run Inference on the Resized Image
        # We pass the numpy array directly to predict()
        self.model.predict(img_resized, verbose=False, device=self.device)
        
        handle.remove()

        if layer_name not in activation:
            return

        # 4. Process Heatmap
        feature_map = activation[layer_name].cpu()
        heatmap = torch.mean(feature_map, dim=1).squeeze().numpy()
        
        # If the heatmap has negative values (inhibition), 
        # shift everything up so the lowest value starts at 0.
        min_val = np.min(heatmap)
        if min_val < 0:
            heatmap -= min_val

        # Normalize (0 to 1)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # 5. Resize Heatmap Back to Original
        heatmap_final = cv2.resize(heatmap, (orig_w, orig_h))

        # Dynamic blur based on image width
        blur_radius = int(orig_w / 25) # Slightly reduced blur for sharper details
        if blur_radius % 2 == 0: blur_radius += 1
        heatmap_final = cv2.GaussianBlur(heatmap_final, (blur_radius, blur_radius), 0)

        # 6. Overlay
        heatmap_uint8 = np.uint8(255 * heatmap_final)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Blend
        superimposed_img = cv2.addWeighted(img_raw, 0.6, heatmap_color, 0.4, 0)

        # 7. Save
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_name = f"{os.path.splitext(filename)[0]}_{layer_name.replace('.', '_')}.jpg"
        save_path = os.path.join(output_dir, save_name)
        
        cv2.imwrite(save_path, superimposed_img)
        print(f"Saved: {save_path}")