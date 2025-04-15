import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

# Paths
clean_dir = "../my_dataset/celeba_128x128_20000/clean"
noisy_dir = "../my_dataset/celeba_128x128_20000/noisy"
model_base_path = "models"

# Model configs
model_names = [
    "DnCNN_mae_e10_b16_v1",
    "UDnCNN_mae_e10_b16_v1",
    "DnCNN_mae_e25_b16_v1",
    "UDnCNN_mae_e25_b16_v1",
    "DnCNN_mae_e50_b16_v1",
    "UDnCNN_mae_e50_b16_v1",
]

# Get sorted filenames (assumes both dirs have matching filenames)
filenames = sorted(os.listdir(clean_dir))
assert filenames == sorted(os.listdir(noisy_dir)), "Mismatch between clean and noisy files."

# Pick a random test image
rand_file = random.choice(filenames)
print(f"🖼️ Selected image: {rand_file}")

# Load and normalize an image
def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

# Load input and clean image
input_image = load_image(os.path.join(noisy_dir, rand_file))[np.newaxis, ...]  # (1, H, W, C)
clean_image = load_image(os.path.join(clean_dir, rand_file))

# Inference
outputs = {}

for model_name in model_names:
    model_path = os.path.join(model_base_path, model_name, "model.keras")
    if not os.path.exists(model_path):
        print(f"⚠️ Missing model: {model_path}")
        continue

    print(f"🔍 Loading model: {model_name}")
    model = load_model(model_path, compile=False)
    prediction = model.predict(input_image, verbose=0)[0]
    outputs[model_name] = prediction

# Plotting
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

# Display noisy input and ground truth
axes[0].imshow(np.clip(input_image[0], 0, 1))
axes[0].set_title("Noisy Input")
axes[0].axis("off")

axes[1].imshow(np.clip(clean_image, 0, 1))
axes[1].set_title("Ground Truth")
axes[1].axis("off")

# Show model outputs
for i, model_name in enumerate(model_names):
    if model_name in outputs:
        ax = axes[i + 2]
        ax.imshow(np.clip(outputs[model_name], 0, 1))
        ax.set_title(model_name)
        ax.axis("off")

# Hide unused axes if any
for j in range(len(model_names) + 2, len(axes)):
    axes[j].axis("off")

output_path = "inference_comparison.png"  # or "inference_comparison.pdf" for vector format
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {output_path}")
plt.tight_layout()
plt.show()