import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ------------------------------
# Configuration
# ------------------------------
clean_dir = "../my_dataset/celeba_256x256_10000/clean"
noisy_dir = "../my_dataset/celeba_256x256_10000/noisy"
models_dir = "./models"
output_img_dir = "./results/loss_comparison_256x256"

os.makedirs(output_img_dir, exist_ok=True)

model_names = [
    "UDnCNN_mae_e2000_b16_v2",
    "UDnCNN_mse_e2000_b16_v2",
    "UDnCNN_ssim_mse_a0.3_e2000_b16_v2",
    "UDnCNN_ssim_mse_a0.84_e2000_b16_v2",
    "UDnCNN_ssim_mse_a0_e2000_b16_v2",
    "UDnCNN_vgg_mse_a0.3_e2000_b16_v2",
    "UDnCNN_vgg_mse_a0.84_e2000_b16_v2",
    "UDnCNN_vgg_mse_a0_e2000_b16_v2",
]

# ------------------------------
# Load and normalize a random image
# ------------------------------
filenames = sorted(os.listdir(clean_dir))
assert filenames == sorted(os.listdir(noisy_dir)), "Mismatch between clean and noisy datasets."

rand_file = random.choice(filenames)
print(f"🖼️ Selected image: {rand_file}")

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0

noisy_image = load_image(os.path.join(noisy_dir, rand_file))
clean_image = load_image(os.path.join(clean_dir, rand_file))

# ------------------------------
# Save input and ground truth
# ------------------------------
def save_image(img_array, filename):
    img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
    img.save(os.path.join(output_img_dir, filename))

save_image(noisy_image, "noisy_input.png")
save_image(clean_image, "ground_truth.png")

# ------------------------------
# Predict using patch-wise inference
# ------------------------------
def predict_in_patches(model, image, patch_size=64):
    h, w, c = image.shape
    assert h == w == 256, "Input image must be 256x256"

    output = np.zeros_like(image)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_input = np.expand_dims(patch, axis=0)
            denoised_patch = model.predict(patch_input, verbose=0)[0]
            output[i:i+patch_size, j:j+patch_size] = denoised_patch
    return output

# ------------------------------
# Process all models and save outputs
# ------------------------------
for model_name in model_names:
    model_path = os.path.join(models_dir, model_name, "model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Skipping missing model: {model_path}")
        continue

    print(f"🔍 Running inference with: {model_name}")
    model = load_model(model_path, compile=False)
    prediction = predict_in_patches(model, noisy_image)
    
    short_name = model_name.replace("_e2000_b16_v2", "").replace("UDnCNN_", "").replace("_mse", "")
    out_filename = f"{short_name}.png"
    save_image(prediction, out_filename)