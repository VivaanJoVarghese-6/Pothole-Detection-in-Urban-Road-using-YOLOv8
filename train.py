"""
Pothole Detection using YOLOv8 - Training Script
Task 20: Pothole Detection in Urban Roads
Dataset: Pothole Segmentation YOLOv8 - Roboflow
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "model":      "yolov8n-seg.pt",   # nano segmentation model (lightweight)
    "data_yaml":  "data/data.yaml",   # path to dataset config
    "epochs":     50,
    "imgsz":      640,
    "batch":      16,
    "lr0":        0.01,
    "patience":   15,                 # early stopping patience
    "project":    "runs/train",
    "name":       "pothole_seg",
    "device":     "0",                # GPU id; use 'cpu' if no GPU
    "workers":    4,
    "conf_thres": 0.25,
    "iou_thres":  0.45,
    "augment":    True,
}


# ─── Dataset YAML Generator ───────────────────────────────────────────────────

def create_dataset_yaml(data_dir: str = "data") -> str:
    """
    Create data.yaml if it doesn't exist (fallback for manual dataset setup).
    Roboflow exports already include this file — only used if missing.
    """
    yaml_path = os.path.join(data_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"[INFO] Found existing data.yaml at {yaml_path}")
        return yaml_path

    dataset_config = {
        "path":  os.path.abspath(data_dir),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["pothole"],
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"[INFO] Created data.yaml at {yaml_path}")
    return yaml_path


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    """Train YOLOv8 segmentation model on the pothole dataset."""

    print("=" * 60)
    print("  Pothole Detection — YOLOv8 Training")
    print("=" * 60)

    # Verify dataset
    data_yaml = create_dataset_yaml()
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            "Dataset not found. Please download from Roboflow:\n"
            "https://universe.roboflow.com/pothole-detection-system/"
            "pothole-segmentation-yolov8\n"
            "and extract into the 'data/' folder."
        )

    # Load model
    print(f"\n[INFO] Loading model: {CONFIG['model']}")
    model = YOLO(CONFIG["model"])

    # Train
    print(f"[INFO] Starting training for {CONFIG['epochs']} epochs...")
    results = model.train(
        data       = data_yaml,
        epochs     = CONFIG["epochs"],
        imgsz      = CONFIG["imgsz"],
        batch      = CONFIG["batch"],
        lr0        = CONFIG["lr0"],
        patience   = CONFIG["patience"],
        project    = CONFIG["project"],
        name       = CONFIG["name"],
        device     = CONFIG["device"],
        workers    = CONFIG["workers"],
        augment    = CONFIG["augment"],
        save       = True,
        plots      = True,
        verbose    = True,
    )

    print("\n[INFO] Training complete!")
    best_weights = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
    print(f"[INFO] Best weights saved at: {best_weights}")
    return str(best_weights)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(weights_path: str = None):
    """Validate the trained model on the test set."""

    if weights_path is None:
        weights_path = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"

    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found at {weights_path}. Train first.")
        return

    print(f"\n[INFO] Validating model: {weights_path}")
    model = YOLO(weights_path)

    metrics = model.val(
        data    = CONFIG["data_yaml"],
        imgsz   = CONFIG["imgsz"],
        batch   = CONFIG["batch"],
        conf    = CONFIG["conf_thres"],
        iou     = CONFIG["iou_thres"],
        project = "runs/val",
        name    = CONFIG["name"],
        plots   = True,
    )

    print("\n─── Validation Metrics ───────────────────────────")
    print(f"  mAP50      : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95   : {metrics.seg.map:.4f}")
    print(f"  Precision  : {metrics.seg.mp:.4f}")
    print(f"  Recall     : {metrics.seg.mr:.4f}")
    print("──────────────────────────────────────────────────")
    return metrics


# ─── Plot Training Curves ─────────────────────────────────────────────────────

def plot_training_results(results_csv: str = None):
    """
    Plot training loss and metric curves from results.csv
    Generated automatically by Ultralytics, but this gives a clean custom plot.
    """
    import pandas as pd

    if results_csv is None:
        results_csv = f"{CONFIG['project']}/{CONFIG['name']}/results.csv"

    if not os.path.exists(results_csv):
        print("[WARN] results.csv not found, skipping plot.")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("YOLOv8 Pothole Detection — Training Results", fontsize=14, fontweight="bold")

    # Loss curves
    if "train/seg_loss" in df.columns:
        axes[0, 0].plot(df["epoch"], df["train/seg_loss"], label="Train Seg Loss", color="#185FA5")
        axes[0, 0].plot(df["epoch"], df["val/seg_loss"],   label="Val Seg Loss",   color="#D85A30")
        axes[0, 0].set_title("Segmentation Loss")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Epoch")

    if "train/box_loss" in df.columns:
        axes[0, 1].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="#185FA5")
        axes[0, 1].plot(df["epoch"], df["val/box_loss"],   label="Val Box Loss",   color="#D85A30")
        axes[0, 1].set_title("Box Loss")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Epoch")

    if "metrics/mAP50(B)" in df.columns:
        axes[1, 0].plot(df["epoch"], df["metrics/mAP50(B)"],    label="Box mAP50",    color="#1D9E75")
        axes[1, 0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="Box mAP50-95", color="#639922")
        axes[1, 0].set_title("Box Detection mAP")
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("Epoch")

    if "metrics/mAP50(M)" in df.columns:
        axes[1, 1].plot(df["epoch"], df["metrics/mAP50(M)"],    label="Mask mAP50",    color="#7F77DD")
        axes[1, 1].plot(df["epoch"], df["metrics/mAP50-95(M)"], label="Mask mAP50-95", color="#D4537E")
        axes[1, 1].set_title("Segmentation Mask mAP")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Epoch")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{CONFIG['project']}/{CONFIG['name']}/custom_training_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Training plot saved: {save_path}")
    plt.show()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    weights = train()
    validate(weights)
    plot_training_results()
