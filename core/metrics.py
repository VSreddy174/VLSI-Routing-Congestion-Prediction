import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred, mask, is_binary=False):
    """
    Computes a comprehensive suite of metrics for a single sample.
    Works for both regression and classification outputs.
    """
    # Flatten and filter by mask to only evaluate active chip area
    t_f = y_true[mask == 1].flatten()
    p_f = y_pred[mask == 1].flatten()

    # --- 1. General Spatial Metrics ---
    rmse = np.sqrt(np.mean((t_f - p_f)**2))
    mae = np.mean(np.abs(t_f - p_f))
    
    # SSIM requires the full 2D grid
    ssim = structural_similarity(y_true, y_pred, data_range=1.0)
    
    # Pearson Correlation Coefficient
    if np.std(t_f) > 0 and np.std(p_f) > 0:
        pcc = pearsonr(t_f, p_f)[0]
    else:
        pcc = 0.0

    # --- 2. Classification/Hotspot Metrics ---
    # For metrics like F1, we need binary labels
    if not is_binary:
        # If it was a regression task, we threshold at 0.1 to define a 'hotspot'
        t_b = (t_f > 0.1).astype(int)
        p_b = (p_f > 0.1).astype(int)
    else:
        t_b = t_f.astype(int)
        p_b = (p_f > 0.5).astype(int)

    # AUC and PR-AUC
    if len(np.unique(t_b)) > 1:
        auc = roc_auc_score(t_b, p_f)
        pr_auc = average_precision_score(t_b, p_f)
    else:
        auc, pr_auc = 0.5, 0.0

    # F1, Precision, Recall, and MCC
    precision, recall, f1, _ = precision_recall_fscore_support(t_b, p_b, average='binary', zero_division=0)
    mcc = matthews_corrcoef(t_b, p_b)

    return {
        "RMSE": rmse, "MAE": mae, "PCC": pcc, "SSIM": ssim,
        "AUC": auc, "PR-AUC": pr_auc, "F1": f1, "MCC": mcc
    }

class MetricTracker:
    """Helper class to accumulate metrics across a full test set."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.results_h = []
        self.results_v = []

    def update(self, y_true_batch, y_pred_batch, mask_batch, is_binary=False):
        # Move to CPU and numpy
        y_t = y_true_batch.cpu().numpy()
        y_p = y_pred_batch.cpu().numpy()
        m = mask_batch.cpu().numpy()

        for i in range(len(y_t)):
            # Horizontal Metrics (Index 0)
            self.results_h.append(calculate_metrics(y_t[i, 0], y_p[i, 0], m[i], is_binary))
            # Vertical Metrics (Index 1)
            self.results_v.append(calculate_metrics(y_t[i, 1], y_p[i, 1], m[i], is_binary))

    def get_averages(self):
        avg_h = {k: np.mean([x[k] for x in self.results_h]) for k in self.results_h[0].keys()}
        avg_v = {k: np.mean([x[k] for x in self.results_v]) for k in self.results_v[0].keys()}
        return avg_h, avg_v
