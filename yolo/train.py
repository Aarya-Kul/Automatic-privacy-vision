from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import pandas as pd

hyperparameters = {
    "epochs": 50,
    "img_size": 640,
    "batch_size": 16,
    "learning_rate": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "workers": 4,
}

model = YOLO("yolo11n-seg.yaml")

model.train(
    data="../../image_privacy_data/multiclass_data.yaml",
    epochs=hyperparameters["epochs"],
    imgsz=hyperparameters["img_size"],
    batch=hyperparameters["batch_size"],
    lr0=hyperparameters["learning_rate"],
    momentum=hyperparameters["momentum"],
    weight_decay=hyperparameters["weight_decay"],
    device=hyperparameters["device"],
    workers=hyperparameters["workers"],
)

metrics = model.val()

print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")

# Get the list of files in the directory
files = list(Path("../runs/segment").iterdir())

# Sort the files by creation time in descending order
files_sorted_by_ctime = sorted(files, key=lambda f: f.stat().st_ctime, reverse=True)

# Check if there are at least two files
assert len(files_sorted_by_ctime) >= 2, "there may not be a results dir for training"

latest_training_dir = files_sorted_by_ctime[1]

print(
    f"getting results for the most recently created training results directory: {latest_training_dir}"
)

results_path = f"{latest_training_dir}/results.csv"

results = pd.read_csv(results_path)


# dont know if this will work just GPTed plots
# Plot training loss curves for Box Loss, Class Loss, and Object Loss
def plot_losses(results, run):
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))

    # Plot each loss type
    plt.plot(results["epoch"], results["val/box_loss"], label="Box Loss", color="blue")
    plt.plot(
        results["epoch"], results["val/cls_loss"], label="Class Loss", color="green"
    )
    plt.plot(
        results["epoch"], results["val/dfl_loss"], label="Object Loss", color="red"
    )

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.savefig(f"{run}_loss_fig.png")


# Plot the training loss
plot_losses(results, latest_training_dir.name)


# Plot the evaluation metrics: mAP, Precision, Recall, etc.
def plot_metrics(metrics, run):
    # Define the labels and their corresponding values
    labels = ["mAP@50", "mAP@50-95", "Avg. Precision", "Avg. Recall"]
    values = [
        metrics.box.map50,
        metrics.box.map,
        np.mean(metrics.box.p),
        np.mean(metrics.box.r),
    ]

    # Create the barplot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=values, palette="viridis")

    # Add title and adjust y-axis
    plt.title("YOLO Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Metrics typically range from 0 to 1
    plt.savefig(f"{run}_metrics_fig.png")


# Plot the evaluation metrics
plot_metrics(metrics, latest_training_dir.name)
