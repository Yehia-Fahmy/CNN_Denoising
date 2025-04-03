import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Constants ===
IMAGE_DIR = "/home/yehia/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
NEW_DIR = "/home/yehia/celaba_datasets"
IMAGE_SIZE = (128, 128)
BLUR_KERNEL = (5, 5)
NOISE_MEAN = 0
NOISE_STDDEV = 25
MAX_IMAGES = 20000

def get_image_paths(directory, max_images=None):
    print(f"📁 Scanning directory: {directory}")
    image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith(".jpg")]
    image_paths = sorted(image_paths)  # Optional: sort alphabetically
    if max_images:
        image_paths = image_paths[:max_images]
    print(f"🔍 Found {len(image_paths)} image files to process.")
    return image_paths

def preprocess_and_save(image_path, base_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Warning: Unable to load image: {image_path}")
        return

    img_resized = cv2.resize(img, IMAGE_SIZE)

    # Noisy
    img_blurred = cv2.GaussianBlur(img_resized, BLUR_KERNEL, 0)
    img_float = img_blurred.astype(np.float32) / 255.0
    noise = np.random.normal(NOISE_MEAN / 255.0, NOISE_STDDEV / 255.0, img_float.shape).astype(np.float32)
    img_noisy = np.clip(img_float + noise, 0.0, 1.0)
    img_noisy_uint8 = (img_noisy * 255).astype(np.uint8)

    # Clean (resized)
    img_clean_uint8 = img_resized

    # Save paths
    dir_name, base_name = os.path.split(image_path)
    noisy_dir = os.path.join(base_dir, "noisy")
    clean_dir = os.path.join(base_dir, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    cv2.imwrite(os.path.join(noisy_dir, base_name), img_noisy_uint8)
    cv2.imwrite(os.path.join(clean_dir, base_name), img_clean_uint8)

    print(f"💾 Saved: {base_name}")

def main():
    image_paths = get_image_paths(IMAGE_DIR, max_images=MAX_IMAGES)
    base_dir = f"{NEW_DIR}/celeba_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_{len(image_paths)}"
    os.makedirs(NEW_DIR, exist_ok=True)
    print(f"🗂️ Saving to: {base_dir}")

    # Pass both path and output dir to executor
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(preprocess_and_save, path, base_dir) for path in image_paths]
        for future in as_completed(futures):
            future.result()

    print("✅ Parallel image processing complete.")

if __name__ == "__main__":
    main()
