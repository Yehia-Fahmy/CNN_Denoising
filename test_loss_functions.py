import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import math

# -----------------------------------
# CONFIGURATION
# -----------------------------------
clean_dir = "../my_dataset/celeba_64x64_20000/clean"
noisy_dir = "../my_dataset/celeba_64x64_20000/noisy"
models_dir = "./models"
output_img_dir = "./results/loss_comparison_64x64"

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

# -----------------------------------
# Load a random image from the dataset
# -----------------------------------
filenames = sorted(os.listdir(clean_dir))
assert filenames == sorted(os.listdir(noisy_dir)), "Mismatch between clean and noisy datasets."

rand_file = random.choice(filenames)
print(f"🖼️ Selected image: {rand_file}")

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0

def save_image(img_array, filename):
    img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
    img.save(os.path.join(output_img_dir, filename))

noisy_image = load_image(os.path.join(noisy_dir, rand_file))
clean_image = load_image(os.path.join(clean_dir, rand_file))
input_image = np.expand_dims(noisy_image, axis=0)

# Save input and ground truth
save_image(noisy_image, "noisy_input.png")
save_image(clean_image, "ground_truth.png")

# -----------------------------------
# Run inference and save individual images
# -----------------------------------
outputs = {}

for model_name in model_names:
    model_path = os.path.join(models_dir, model_name, "model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        continue

    print(f"🔍 Running inference: {model_name}")
    model = load_model(model_path, compile=False)
    prediction = model.predict(input_image, verbose=0)[0]
    outputs[model_name] = prediction

    # Generate filename-friendly label
    label = model_name.replace("_e2000_b16_v2", "").replace("UDnCNN_", "").replace("_mse", "")
    save_image(prediction, f"{label}.png")

# -----------------------------------
# Generate summary plot with all images
# -----------------------------------
num_cols = 5
num_images = 2 + len(outputs)
num_rows = math.ceil(num_images / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
axes = axes.flatten()

# Noisy and clean
axes[0].imshow(np.clip(noisy_image, 0, 1))
axes[0].set_title("Noisy Input")
axes[0].axis("off")

axes[1].imshow(np.clip(clean_image, 0, 1))
axes[1].set_title("Ground Truth")
axes[1].axis("off")

# Model predictions
for i, (model_name, pred) in enumerate(outputs.items(), start=2):
    label = model_name.replace("_e2000_b16_v2", "").replace("UDnCNN_", "").replace("_mse", "")
    axes[i].imshow(np.clip(pred, 0, 1))
    axes[i].set_title(label)
    axes[i].axis("off")

# Hide extra axes
for j in range(num_images, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
summary_path = os.path.join(output_img_dir, "inference_comparison_e2000_multiline.png")
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
print(f"\n✅ Saved full figure to: {summary_path}")
plt.show()