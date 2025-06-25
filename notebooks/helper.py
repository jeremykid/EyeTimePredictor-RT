import torch
from typing import Tuple, List
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from Transformer_classify_model import TransformerClassifier
from coral_pytorch.losses import corn_loss
from multi_class_dataloader import TimeMultiClassDataset, collate_time_series    
import copy
from torch.nn import CrossEntropyLoss
from coral_pytorch.dataset import corn_label_from_logits

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Optional, Union

def validate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[List[int], List[int], List[List[float]]]:
    """
    validate model，return ground truth, pred label, pred prob

    Args:
        model: load PyTorch model，forward return logits。
        loader: DataLoader，batch include (padded_X, lengths, labels) 
        device: torch.device，'cuda' or 'cpu'。

    Returns:
        label_list: List[int], all samples ground truth
        pred_list:  List[int], all samples pred label
        proba_list: List[List[float]], all samples pred prob
    example:
        label_list, pred_list, proba_list = validate_model(model, val_loader, device)
    """
    model.eval()
    label_list: List[int] = []
    pred_list:  List[int] = []
    proba_list: List[List[float]] = []

    with torch.no_grad():
        for padded_X, lengths, labels in loader:
            # 1) normalize
            padded_X = padded_X.to(device)
            mean = padded_X.mean(dim=(0,2), keepdim=True)
            std  = padded_X.std(dim=(0,2), keepdim=True) + 1e-6
            padded_X = (padded_X - mean) / std

            # 2) prepare Transformer input (batch, feat, T) -> (T, batch, feat)
            padded_X = padded_X.permute(2, 0, 1)
            seq_len, batch_size, _ = padded_X.shape
            mask = torch.arange(seq_len, device=device) \
                       .unsqueeze(1) >= lengths.to(device).unsqueeze(0)
            src_key_padding_mask = mask.T  # (batch, seq_len)

            # 3) forward calculate logits
            logits = model(padded_X, src_key_padding_mask)

            # 4) calculate prob and pred label
            proba = torch.softmax(logits, dim=1) \
                         .cpu().tolist()               # List[List[float]]
            preds = torch.argmax(logits, dim=1) \
                         .cpu().tolist()               # List[int]
            trues = labels.tolist()                     # List[int]

            # 5) get results
            proba_list.extend(proba)
            pred_list.extend(preds)
            label_list.extend(trues)

    return label_list, pred_list, proba_list

def plot_and_save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = None,
    display_labels: Optional[List[str]] = None,
    normalize: Optional[Union[None, str]] = None,
    cmap: str = 'Blues',
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    dpi: int = 100
) -> pd.DataFrame:
    """
    Compute, plot, and optionally save a confusion matrix.

    Args:
        y_true:         True class labels.
        y_pred:         Predicted class labels.
        labels:         Label values to index the matrix (e.g. [0,1,2,3]).
        display_labels: Names shown on axes (e.g. ['1','2','3','4']).
        normalize:      One of {None, 'true', 'pred', 'all'}.
        cmap:           Matplotlib colormap.
        title:          Title of the plot.
        save_path:      If given, saves the figure to this path.
        figsize:        Figure size.
        dpi:            Resolution for saved figure.

    Returns:
        A pandas DataFrame of the confusion matrix.
        
    Example:    
        cm_df = plot_and_save_confusion_matrix(
        y_true=label_list,
        y_pred=pred_list,
        labels=[0,1,2,3],
        display_labels=['1','2','3','4'],
        normalize=None,
        cmap='Blues',
        title="Multi-Class Confusion Matrix",
        save_path="confusion_matrix.png"  # or omit to skip saving
    )
    """
    # 1. Compute (and normalize) the confusion matrix array
    cm_array = confusion_matrix(
        y_true, y_pred,
        labels=labels,
        normalize=normalize  # normalize rows/cols/whole matrix here :contentReference[oaicite:0]{index=0}
    )

    # 2. Build DataFrame for CSV export & analysis
    if display_labels is None and labels is not None:
        disp_labels = [str(l) for l in labels]
    else:
        disp_labels = display_labels

    cm_df = pd.DataFrame(cm_array, index=disp_labels, columns=disp_labels)

    # 3. Plot with ConfusionMatrixDisplay (no normalize arg here!)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=disp_labels)
    disp.plot(
        ax=ax,
        cmap=cmap,
        colorbar=True,
        xticks_rotation='horizontal'  # rotation option from skl docs :contentReference[oaicite:1]{index=1}
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()

    # 4. Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
    return cm_df

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

def top_k_accuracy(y_true, y_proba, k=2):
    """
    Compute Top-K accuracy: fraction of samples where true label is in the model's top-K probabilities.
    """
    topk = np.argsort(y_proba, axis=1)[:, -k:]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])

def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    class_labels=None,
    top_k=2
):
    """
    Compute a variety of multi-class evaluation metrics.

    Args:
        y_true (array-like): Ground-truth integer labels, shape (n_samples,).
        y_pred (array-like): Predicted integer labels, shape (n_samples,).
        y_proba (array-like, optional): Predicted probabilities,
            shape (n_samples, n_classes). Required for log_loss and ROC-AUC.
        class_labels (list, optional): List of class names or integers
            to use in reports. If None, will infer sorted unique labels.
        top_k (int): K for Top-K accuracy calculation.

    Returns:
        metrics_dict (dict): Raw metric values.
        report_str (str): Formatted text report.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine classes
    classes = np.unique(y_true) if class_labels is None else class_labels

    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)

    # 2. Precision / Recall / F1 (macro, micro, weighted)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # 3. Detailed classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=[str(c) for c in classes],
        digits=4,
        zero_division=0
    )

    # 4. Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # 5. Cohen’s Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # 6. Log-Loss (requires probabilities)
    ll = None
    if y_proba is not None:
        ll = log_loss(y_true, y_proba)

    # 7. ROC-AUC (One-vs-Rest)
    roc_auc = None
    if y_proba is not None:
        y_onehot = label_binarize(y_true, classes=classes)
        roc_auc = roc_auc_score(y_onehot, y_proba, average='macro', multi_class='ovr')

    # 8. Top-K Accuracy
    topk_acc = None
    if y_proba is not None:
        topk_acc = top_k_accuracy(y_true, y_proba, k=top_k)

    # Assemble results
    metrics = {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'precision_micro': prec_micro,
        'recall_micro': rec_micro,
        'f1_micro': f1_micro,
        'precision_weighted': prec_weight,
        'recall_weighted': rec_weight,
        'f1_weighted': f1_weight,
        'mcc': mcc,
        'kappa': kappa,
        'log_loss': ll,
        'roc_auc': roc_auc,
        f'top_{top_k}_accuracy': topk_acc,
    }

    # Build the textual report
    lines = [
        f"Accuracy:           {acc:.4f}",
        f"Macro  P/R/F1:      {prec_macro:.4f}/{rec_macro:.4f}/{f1_macro:.4f}",
        f"Micro  P/R/F1:      {prec_micro:.4f}/{rec_micro:.4f}/{f1_micro:.4f}",
        f"Weighted P/R/F1:    {prec_weight:.4f}/{rec_weight:.4f}/{f1_weight:.4f}",
        f"MCC:                {mcc:.4f}",
        f"Cohen’s Kappa:      {kappa:.4f}"
    ]
    if ll is not None:
        lines.append(f"Log-Loss:           {ll:.4f}")
    if roc_auc is not None:
        lines.append(f"ROC-AUC (OvR):      {roc_auc:.4f}")
    if topk_acc is not None:
        lines.append(f"Top-{top_k} Accuracy:     {topk_acc:.4f}")
    lines.append("\nClassification Report:")
    lines.append(report)

    report_str = "\n".join(lines)
    return metrics, report_str


def train_model(train_loader, val_loader, class_counts, device, pooling_method = 'attention'):
    NUM_CLASSES = 4
    model = TransformerClassifier(
        feature_dim=6, d_model=64, nhead=8, num_layers=3,
        dim_feedforward=256, dropout=0.1, num_classes=NUM_CLASSES,
        pooling=pooling_method
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # define EarlyStopping 
    patience = 5              
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    # loss_fn = corn_loss
    # class_counts = [114, 90, 89, 35]
    inv_freq = [1.0/x for x in class_counts]
    weights = torch.tensor(inv_freq) / sum(inv_freq)  # 归一化
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    # criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for padded_X, lengths, labels in train_loader:
            padded_X, lengths, labels = (
                padded_X.to(device), lengths.to(device), labels.to(device)
            )
            mean = padded_X.mean(dim=(0,2), keepdim=True)
            std  = padded_X.std(dim=(0,2), keepdim=True) + 1e-6
            padded_X = (padded_X - mean) / std

            padded_X = padded_X.permute(2,0,1)
            seq_len = padded_X.size(0)
            mask = torch.arange(seq_len, device=device).unsqueeze(1) >= lengths.unsqueeze(0)
            src_key_padding_mask = mask.T

            logits = model(padded_X, src_key_padding_mask)
            # loss = loss_fn(logits, labels, NUM_CLASSES)
            loss = criterion(logits.to(torch.float64), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for padded_X, lengths, labels in val_loader:
                padded_X, lengths, labels = (
                    padded_X.to(device), lengths.to(device), labels.to(device)
                )
                mean = padded_X.mean(dim=(0,2), keepdim=True)
                std  = padded_X.std(dim=(0,2), keepdim=True) + 1e-6
                padded_X = (padded_X - mean) / std
                padded_X = padded_X.permute(2,0,1)
                seq_len = padded_X.size(0)
                mask = torch.arange(seq_len, device=device).unsqueeze(1) >= lengths.unsqueeze(0)
                src_key_padding_mask = mask.T

                logits = model(padded_X, src_key_padding_mask)
                # val_losses.append(corn_loss(logits, labels, NUM_CLASSES).item())
                val_losses.append(criterion(logits.to(torch.float64), labels).item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  > No improvement for {epochs_no_improve}/{patience} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Restoring best model from epoch {epoch+1-patience}.")
            model.load_state_dict(best_model_state)
            break
    return model

