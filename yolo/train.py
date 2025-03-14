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
}

model = YOLO("yolov8s.yaml")

model.train(
    data="dataset.yaml",
    epochs=hyperparameters["epochs"],
    imgsz=hyperparameters["img_size"],
    batch=hyperparameters["batch_size"],
    lr0=hyperparameters["learning_rate"],
    momentum=hyperparameters["momentum"],
    weight_decay=hyperparameters["weight_decay"],
    device=hyperparameters["device"],
)

metrics = model.val()

print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.recall}")

results_path = "runs/train/exp*/results.csv"

results = pd.read_csv(results_path)


# dont know if this will work just GPTed plots
# Plot training loss curves for Box Loss, Class Loss, and Object Loss
def plot_losses(results):
    sns.set(style="darkgrid")

    plt.figure(figsize=(12, 6))

    # Plot each loss type
    plt.plot(results["epoch"], results["box_loss"], label="Box Loss", color="blue")
    plt.plot(results["epoch"], results["cls_loss"], label="Class Loss", color="green")
    plt.plot(results["epoch"], results["obj_loss"], label="Object Loss", color="red")

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.show()


# Plot the training loss
plot_losses(results)


# Plot the evaluation metrics: mAP, Precision, Recall, etc.
def plot_metrics(metrics):
    # Define the labels and their corresponding values
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    values = [
        metrics.box.map50,
        metrics.box.map,
        metrics.box.precision,
        metrics.box.recall,
    ]

    # Create the barplot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=values, palette="viridis")

    # Add title and adjust y-axis
    plt.title("YOLO Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Metrics typically range from 0 to 1
    plt.show()


# Plot the evaluation metrics
plot_metrics(metrics)
