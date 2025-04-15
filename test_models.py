import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------------
# CONFIGURATION
# -----------------------------------
clean_dir = "../my_dataset/celeba_128x128_20000/clean"
noisy_dir = "../my_dataset/celeba_128x128_20000/noisy"
model_base_path = "models"
output_img_dir = "./results/dncnn_udncnn_mae_comparison"

os.makedirs(output_img_dir, exist_ok=True)

# Model configs
model_names = [
    "DnCNN_mae_e10_b16_v1",
    "UDnCNN_mae_e10_b16_v1",
    "DnCNN_mae_e25_b16_v1",
    "UDnCNN_mae_e25_b16_v1",
    "DnCNN_mae_e50_b16_v1",
    "UDnCNN_mae_e50_b16_v1",
]

# -----------------------------------
# Load random image from dataset
# -----------------------------------
filenames = sorted(os.listdir(clean_dir))
assert filenames == sorted(os.listdir(noisy_dir)), "Mismatch between clean and noisy files."

rand_file = random.choice(filenames)
print(f"🖼️ Selected image: {rand_file}")

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0

def save_image(img_array, filename):
    img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
    img.save(os.path.join(output_img_dir, filename))

# Load and save noisy + clean images
input_image = load_image(os.path.join(noisy_dir, rand_file))
clean_image = load_image(os.path.join(clean_dir, rand_file))

save_image(input_image, "noisy_input.png")
save_image(clean_image, "ground_truth.png")

input_batch = input_image[np.newaxis, ...]  # (1, H, W, C)

# -----------------------------------
# Run inference and save individual outputs
# -----------------------------------
outputs = {}

for model_name in model_names:
    model_path = os.path.join(model_base_path, model_name, "model.keras")
    if not os.path.exists(model_path):
        print(f"⚠️ Missing model: {model_path}")
        continue

    print(f"🔍 Loading model: {model_name}")
    model = load_model(model_path, compile=False)
    prediction = model.predict(input_batch, verbose=0)[0]
    outputs[model_name] = prediction

    # Clean name for saving
    label = model_name.replace("DnCNN_", "dncnn_").replace("UDnCNN_", "udncnn_")
    save_image(prediction, f"{label}.png")

# -----------------------------------
# Create summary figure
# -----------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

axes[0].imshow(np.clip(input_image, 0, 1))
axes[0].set_title("Noisy Input")
axes[0].axis("off")

axes[1].imshow(np.clip(clean_image, 0, 1))
axes[1].set_title("Ground Truth")
axes[1].axis("off")

for i, model_name in enumerate(model_names):
    if model_name in outputs:
        ax = axes[i + 2]
        ax.imshow(np.clip(outputs[model_name], 0, 1))
        ax.set_title(model_name)
        ax.axis("off")

for j in range(len(model_names) + 2, len(axes)):
    axes[j].axis("off")

summary_path = os.path.join(output_img_dir, "inference_comparison_grid.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Summary figure saved to: {summary_path}")
plt.show()