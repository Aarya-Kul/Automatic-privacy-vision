from pathlib import Path
import psutil
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


msg = """
    Script for running YOLO training. It default to yolo11n-seg.yaml
    if no model is specified. To specify YOLO model,
    pass positional argument, e.g. "yolo11m.pt".
    To specify destination, pass another positional argument with path.
    Note that results are still in ./runs.
    """

parser = argparse.ArgumentParser(description=msg)
parser.add_argument("model", nargs="?", default="yolo11n-seg.yaml")
parser.add_argument("dir_name", nargs="?", default=None)
args = parser.parse_args()
yolo_version = args.model
if args.dir_name is None:
    args.dir_name = yolo_version.split(".")[0]

hyperparameters = {
    "epochs": 75, # experiment with 50 epochs
    "img_size": 640,
    "batch_size": 16,
    "learning_rate": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "degrees": 0.25,
    "scale": 0.3,
    "perspective": 0.0001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "workers": 4,
    "project": "./runs/",
    "name": args.dir_name,
}

# increase num_workers for DataLoader if sufficient memory is available
if (psutil.virtual_memory()[0] / 1000 / 1000 / 1000) >= 48:
    hyperparameters["workers"] = 8

print(f"Using {hyperparameters['workers']} workers for DataLoader")

model = YOLO(yolo_version)

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
    project=hyperparameters["project"],
    name=hyperparameters["name"],
    degrees=hyperparameters["degrees"],
    scale=hyperparameters["scale"],
    perspective=hyperparameters["perspective"],
)

metrics = model.val()

print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")

# Get the list of files in the directory
model_dirs = list(Path("./runs").iterdir())
files = []
for model_dir in model_dirs:
    files.extend(list(model_dir.iterdir()))

# Sort the files by creation time in descending order
files_sorted_by_ctime = sorted(files, key=lambda f: f.stat().st_ctime, reverse=True)

# Check if there are at least two files
assert len(files_sorted_by_ctime) >= 2, "there may not be a results dir for training"

latest_training_dir = files_sorted_by_ctime[0]

i = 0
while not (latest_training_dir / "results.csv").exists():
    i += 1
    if i == len(files_sorted_by_ctime):
        print("no results.csv file found in ../runs/segment subdirectory")
        raise FileNotFoundError
    latest_training_dir = files_sorted_by_ctime[i]

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
