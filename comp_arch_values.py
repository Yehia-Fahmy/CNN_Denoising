import os
import json
import pandas as pd

# Path to your models directory
MODEL_DIR = "/home/yehia/CNN_Denoising/models"

# Filter conditions
epoch_targets = ["e10", "e25", "e50"]
arch_labels = {
    "DnCNN": "DnCNN",
    "UDnCNN": "UDnCNN"
}

results = []

# Iterate through folders
for folder in os.listdir(MODEL_DIR):
    if not folder.endswith("b16_v1"):
        continue

    if not any(epoch in folder for epoch in epoch_targets):
        continue
    if "mae" not in folder.lower():
        continue  # Skip non-MAE models
    if "ssim_mae" in folder.lower():
        continue

    model_dir = os.path.join(MODEL_DIR, folder)
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.isfile(metadata_path):
        print(f"⚠️ Skipping missing metadata: {metadata_path}")
        continue

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    model_name = data.get("model_name", folder)
    if "UDnCNN" in folder:
        arch = "UDnCNN"
    elif "DnCNN" in folder:
        arch = "DnCNN"
    else:
        arch = "Unknown"
    epoch_str = next((e for e in epoch_targets if e in folder), "Unknown")
    epochs = int(epoch_str[1:])

    metrics = data.get("evaluation_metrics", {})
    total_time = data.get("training_time_seconds", None)
    time_per_epoch = total_time / epochs if total_time else None

    results.append({
        "Architecture": arch,
        "Epochs": epochs,
        "MAE": metrics.get("mae"),
        "MSE": metrics.get("mse"),
        "PSNR": metrics.get("psnr_avg"),
        "SSIM": metrics.get("ssim_avg"),
        "Total Time (s)": round(total_time, 2) if total_time else None,
        "Time/Epoch (s)": round(time_per_epoch, 2) if time_per_epoch else None,
    })

# Create DataFrame and sort
df = pd.DataFrame(results)
df = df.sort_values(by=["Epochs", "Architecture"])

# Print results
print("\n📊 DnCNN vs. UDnCNN Performance + Training Time (Epochs: 10, 25, 50):\n")
print(df.to_markdown(index=False))