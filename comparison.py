import os
import json
import pandas as pd

# Set your models folder correctly
MODEL_DIR = "/home/yehia/CNN_Denoising/models"

# Desired configurations
target_configs = {
    "mae": "MAE",
    "mse": "MSE",
    "vgg_mse_a0": "VGG α=0",
    "vgg_mse_a0.3": "VGG α=0.3",
    "vgg_mse_a0.84": "VGG α=0.84",
    "ssim_mse_a0": "SSIM α=0",
    "ssim_mse_a0.3": "SSIM α=0.3",
    "ssim_mse_a0.84": "SSIM α=0.84",
}

results = []

# Go through model directories
for dirname in os.listdir(MODEL_DIR):
    if "e2000" not in dirname:
        continue  # only include models trained for 2000 epochs

    model_dir = os.path.join(MODEL_DIR, dirname)
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.isfile(metadata_path):
        print(f"⚠️ Skipping: {metadata_path} not found")
        continue

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    loss = data.get("loss_function", "").lower()
    alpha = round(data.get("alpha", 0), 2)
    model_name = data.get("model_name", dirname)

    # Build config key
    key = None
    if loss == "mae":
        key = "mae"
    elif loss == "mse":
        key = "mse"
    elif loss == "vgg_mse":
        key = f"vgg_mse_a{alpha}"
    elif loss == "ssim_mse":
        key = f"ssim_mse_a{alpha}"

    if key not in target_configs:
        print(f"Skipping untracked config: {key}")
        continue

    metrics = data.get("evaluation_metrics", {})
    results.append({
        "Configuration": target_configs[key],
        "Model Name": model_name,
        "MAE": metrics.get("mae"),
        "MSE": metrics.get("mse"),
        "PSNR": metrics.get("psnr_avg"),
        "SSIM": metrics.get("ssim_avg"),
    })

# Output results
if not results:
    print("❌ No matching results found.")
else:
    df = pd.DataFrame(results)
    df = df.set_index("Configuration")
    df = df.loc[target_configs.values()]  # Preserve row order
    print("\n📊 Comparison Table:\n")
    print(df.to_markdown())
