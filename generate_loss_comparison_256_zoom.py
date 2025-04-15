import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr

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
# Load random image from dataset
# ------------------------------
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
# Run models and save predictions
# ------------------------------
outputs = [("Noisy Input", noisy_image), ("Ground Truth", clean_image)]

for model_name in model_names:
    model_path = os.path.join(models_dir, model_name, "model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Skipping missing model: {model_path}")
        continue

    print(f"🔍 Running inference with: {model_name}")
    model = load_model(model_path, compile=False)
    prediction = predict_in_patches(model, noisy_image)
    
    label = model_name.replace("_e2000_b16_v2", "").replace("UDnCNN_", "").replace("_mse", "")
    outputs.append((label, prediction))
    save_image(prediction, f"{label}.png")

# ------------------------------
# Create figure with zoom insets and PSNR
# ------------------------------
def crop_zoom(img, box):
    x, y, w, h = box
    return img[y:y+h, x:x+w]

import matplotlib.patches as patches

def plot_with_zoom(ax, image, title, crop_box, metric=None):
    ax.imshow(np.clip(image, 0, 1))
    ax.set_title(title, fontsize=9)
    ax.axis('off')

    # Draw red box on the main image
    x, y, w, h = crop_box
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Create inset axes and zoom region
    inset_ax = inset_axes(ax, width="30%", height="30%", loc='lower left', borderpad=1)
    cropped = image[y:y+h, x:x+w]
    inset_ax.imshow(np.clip(cropped, 0, 1))
    inset_ax.axis('off')

    # Connect the inset to the original box
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="red", lw=1)

    # Optionally add PSNR label
    if metric is not None:
        ax.set_xlabel(f"{metric:.2f} dB", fontsize=8)

# ------------------------------
# Final plot (3 rows × 3 columns)
# ------------------------------
n_cols = 3
n_rows = int(np.ceil(len(outputs) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()

crop_box = (96, 96, 64, 64)  # adjust based on content

for i, (label, img) in enumerate(outputs):
    is_model_output = "Ground" not in label and "Noisy" not in label
    score = psnr(clean_image, img) if is_model_output else None

    if is_model_output:
        plot_with_zoom(axes[i], img, label, crop_box, metric=score)

        # --- Save full image with red box ---
        save_path_full = os.path.join(output_img_dir, f"{label}_full.png")
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(save_path_full)

        # --- Save zoomed-in region ---
        x, y, w, h = crop_box
        zoom_crop = img[y:y+h, x:x+w]
        save_path_zoom = os.path.join(output_img_dir, f"{label}_zoom.png")
        Image.fromarray((np.clip(zoom_crop, 0, 1) * 255).astype(np.uint8)).save(save_path_zoom)

    else:
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(label, fontsize=9)
        axes[i].axis('off')

        # Save noisy/ground truth images too
        img_name = label.lower().replace(" ", "_")
        save_path = os.path.join(output_img_dir, f"{img_name}.png")
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(save_path)

# Hide extra axes
for j in range(len(outputs), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
final_path = os.path.join(output_img_dir, "qualitative_grid_with_zoom.png")
plt.savefig(final_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved figure: {final_path}")
plt.show()