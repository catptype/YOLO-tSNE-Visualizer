import os
from PIL import Image, ImageFile

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported input extensions (lowercase)
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "tif"}

def has_alpha_channel(img: Image.Image) -> bool:
    """Return True if image has an alpha/transparency channel."""
    # Common modes with alpha: "LA", "RGBA"
    if "A" in img.getbands():
        return True
    # Paletted images may have 'transparency' info in img.info
    if img.mode == "P" and ("transparency" in img.info or "transparency" in img.encoderinfo):
        return True
    return False

def flatten_alpha_to_white(img: Image.Image) -> Image.Image:
    """
    Composite image onto a white background to remove transparency.
    Returns an RGB Image.
    """
    # Ensure we work with RGBA for correct alpha compositing
    rgba = img.convert("RGBA")
    white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white_bg, rgba)
    return composed.convert("RGB")

def convert_image_to_jpeg(input_path: str, output_path: str, quality: int = 90):
    """
    Convert a single image to JPEG (RGB), replacing transparent background with white.
    Preserves EXIF if present.
    """
    try:
        with Image.open(input_path) as img:
            # If animated (GIF/webp with frames), use first frame
            if getattr(img, "is_animated", False):
                frame = img.convert("RGBA")
                img_to_use = frame
            else:
                img_to_use = img

            # Prepare final RGB image
            if has_alpha_channel(img_to_use):
                final = flatten_alpha_to_white(img_to_use)
            else:
                # Ensure RGB (converts L, P, CMYK, etc.)
                final = img_to_use.convert("RGB")

            # Try to preserve EXIF (if any)
            exif = img.info.get("exif")

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            save_kwargs = {"format": "JPEG", "quality": quality, "optimize": True}
            if exif:
                save_kwargs["exif"] = exif

            final.save(output_path, **save_kwargs)
            return True, None
    except Exception as e:
        return False, e

def convert_folder_to_jpg(input_dir: str, output_dir: str = None, overwrite: bool = False, quality: int = 90):
    """
    Walks input_dir recursively and converts images to JPEG.
    - If output_dir is None and overwrite is True -> replaces original files (careful).
    - If output_dir provided -> mirrored folder structure placed there.
    """
    input_dir = os.path.abspath(input_dir)
    if output_dir:
        output_dir = os.path.abspath(output_dir)
    else:
        if not overwrite:
            raise ValueError("Either provide output_dir or set overwrite=True to replace originals.")

    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_root) if output_dir else root

        for fname in files:
            ext = fname.rsplit(".", 1)
            if len(ext) < 2:
                continue
            extension = ext[-1].lower()
            if extension not in IMAGE_EXTS:
                continue

            src_path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]
            dest_filename = base + ".jpg"
            dest_path = os.path.join(target_root, dest_filename)

            # If overwriting originals in place, remove original AFTER successful save
            temp_output_path = dest_path
            # Create directory if needed
            os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)

            ok, err = convert_image_to_jpeg(src_path, temp_output_path, quality=quality)
            if ok:
                print(f"[OK] {src_path} → {temp_output_path}")
                if overwrite and not output_dir:
                    # Replace original file (original may have different extension)
                    try:
                        if os.path.abspath(temp_output_path) != os.path.abspath(src_path):
                            os.remove(src_path)
                    except Exception as rm_err:
                        print(f"[WARN] couldn't remove original {src_path}: {rm_err}")
            else:
                print(f"[ERROR] {src_path} failed: {err}")

if __name__ == "__main__":
    # Examples:

    # 1) Convert into a separate output folder (recommended)
    # convert_folder_to_jpg("C:/images/input", "C:/images/output", overwrite=False, quality=90)
    convert_folder_to_jpg("DATASET\example", "DATASET\Output", overwrite=True, quality=90)

# Example usage
