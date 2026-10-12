import json
import logging
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

logger = logging.getLogger(__name__)


class ClassifierEvaluator:
    """Evaluates the model, calculates classification metrics, and generates visual artifacts."""

    def __init__(self, model: tf.keras.Model, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.reports_dir = output_dir / "reports"
        self.figures_dir = output_dir / "figures"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        t0 = time.time()
        y_prob = self.model.predict(x_test, batch_size=256)
        inference_time = time.time() - t0

        y_pred = np.argmax(y_prob, axis=1)

        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        clf_dict = classification_report(y_test, y_pred, output_dict=True, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "macro_f1": float(clf_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(clf_dict["weighted avg"]["f1-score"]),
            "inference_time_sec": float(inference_time),
            "ms_per_sample": float((inference_time / len(x_test)) * 1000),
            "classification_report": clf_dict,
        }

        # Save metrics.json
        with open(self.reports_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save markdown report
        self._write_markdown_report(metrics)

        # Plot & Save figures
        self._plot_confusion_matrix(cm)
        self._plot_misclassified(x_test, y_test, y_pred, y_prob)

        return metrics

    def _write_markdown_report(self, metrics: dict) -> None:
        md_path = self.reports_dir / "classification_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# MNIST Classifier Evaluation Report\n\n")
            f.write(f"- **Test Accuracy:** `{metrics['test_accuracy'] * 100:.2f}%`\n")
            f.write(f"- **Test Loss:** `{metrics['test_loss']:.4f}`\n")
            f.write(f"- **Macro F1-Score:** `{metrics['macro_f1']:.4f}`\n")
            f.write(f"- **Inference Speed:** `{metrics['ms_per_sample']:.4f} ms/sample`\n\n")
            f.write("### Per-Class Performance\n\n")
            f.write("| Digit | Precision | Recall | F1-Score | Support |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for digit in range(10):
                d_str = str(digit)
                if d_str in metrics["classification_report"]:
                    data = metrics["classification_report"][d_str]
                    f.write(
                        f"| {digit} | {data['precision']:.4f} | {data['recall']:.4f} | "
                        f"{data['f1-score']:.4f} | {int(data['support'])} |\n"
                    )

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix.png", dpi=300)
        plt.close()

    def _plot_misclassified(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> None:
        mis_idx = np.where(y_true != y_pred)[0]
        if len(mis_idx) == 0:
            return
        selected = mis_idx[:12]
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        for i, idx in enumerate(selected):
            ax = axes[i // 4, i % 4]
            ax.imshow(x[idx].squeeze(), cmap="gray")
            confidence = probs[idx][y_pred[idx]]
            ax.set_title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}\nConf: {confidence:.2f}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "misclassified_examples.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_training_curves(history: tf.keras.callbacks.History, save_path: Path) -> None:
        hist = history.history
        epochs = range(1, len(hist["loss"]) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, hist["loss"], label="Train Loss", marker="o")
        plt.plot(epochs, hist["val_loss"], label="Val Loss", marker="o")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, hist["accuracy"], label="Train Acc", marker="o")
        plt.plot(epochs, hist["val_accuracy"], label="Val Acc", marker="o")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()