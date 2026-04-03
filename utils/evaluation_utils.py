from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationHelper():
    def plot_confusion_matrix(y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

        tn_val, fp_val, fn_val, tp_val = cm.ravel()
        sensitivity = tp_val / (tp_val + fn_val)
        specificity = tn_val / (tn_val + fp_val)
        precision = tp_val / (tp_val + fp_val)
        accuracy = (tp_val + tn_val) / (tp_val + tn_val + fp_val + fn_val)
        f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

        print(f"True Negatives: {tn_val}")
        print(f"False Positives: {fp_val}")
        print(f"False Negatives: {fn_val}")
        print(f"True Positives: {tp_val}")
        print(f"\nSensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
    def plot_roc_curve(y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Area Under the Curve (AUC): {roc_auc:.4f}")
        
    def plot_precision_recall_curve(y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="#3498db", lw=2, label=f"Precision-Recall curve (AUC = {pr_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Area Under the Curve (AUC): {pr_auc:.4f}")