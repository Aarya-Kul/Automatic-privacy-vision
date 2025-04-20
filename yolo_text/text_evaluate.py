from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import pandas as pd

# Get the list of files in the directory
files = list(Path("../runs/text").iterdir())

# Sort the files by creation time in descending order
files_sorted_by_ctime = sorted(files, key=lambda f: f.stat().st_ctime, reverse=True)

# Check if there are at least two files
assert len(files_sorted_by_ctime) >= 2, "there may not be a results dir for training"

latest_training_dir = files_sorted_by_ctime[0]

i = 0
while not (latest_training_dir / "results.csv").exists():
    i += 1
    if i == len(files_sorted_by_ctime):
        print("no results.csv file found in ../runs/text subdirectory")
        raise FileNotFoundError
    latest_training_dir = files_sorted_by_ctime[i]


latest_training_dir = files_sorted_by_ctime[1]

print(
    f"getting results for the most recently created training results directory: {latest_training_dir}"
)

model = YOLO(f"{latest_training_dir}/weights/best.pt")


metrics = model.val(data="../../image_privacy_data/data.yaml")

print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")

# Get the list of files in the directory
files = list(Path("../runs/text").iterdir())

# Sort the files by creation time in descending order
files_sorted_by_ctime = sorted(files, key=lambda f: f.stat().st_ctime, reverse=True)

# Check if there are at least two files
assert len(files_sorted_by_ctime) >= 2, "there may not be a results dir for training"

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
