# test_cls/internal_test_patient_level.py
#
# One script for all inference: evaluates every fold model of a trained
# configuration on the internal test set and/or ECU, and saves
#   - per-fold metrics, confusion matrices and ROC curves
#   - all-fold ROC plot and fold comparison plot
#   - all_predictions_detailed.csv (per patient, per fold, ensemble)
#
# Internal test: data/manifests/<encoder>_test.csv  (one row per patient, 'paths' ';'-separated)
# ECU:           data/manifests/ecu_features_<encoder>.csv (one row per slide, 'path')
# Label: RSHigh_classic (H if RS > 25), same as training.
#
# Usage (from the folder that contains data/):
#   one configuration:
#     python test_cls/internal_test_patient_level.py --model-folder results_patient_level/h-optimus_acmil_<timestamp>
#   everything in results_patient_level (latest run per encoder+MIL):
#     python test_cls/internal_test_patient_level.py --all
#   options: --cohorts internal ecu (default both), --output-root detailed_predictions_patient_level

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import glob
import json
import re

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from utils.classification_evaluator import ClassificationEvaluator
from models.classification_model_updated import (
    MeanPoolingMILClassifier, AttentionMILClassifier, CLAMClassifier, ACMILClassifier
)

INPUT_DIMS = {
    "resnet18": 512, "resnet50": 2048, "conch_fixed": 512, "uni2-h": 1536,
    "virchow2": 1280, "h-optimus": 1536, "musk": 2048, "dinov2": 1024,
}
MANIFESTS = {
    "internal": "data/manifests/{enc}_test.csv",
    "ecu": "data/manifests/ecu_features_{enc}.csv",
}
RS_THRESHOLD = 25          # high risk if RS > 25
N_FOLDS = 5
METRICS = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy',
           'f1_score', 'sensitivity', 'specificity', 'mcc']


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ------------------------------------------------------------------ data
def load_feature_file(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("features", "feats", "x"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(f"No features key in {path}: {list(obj.keys())}")
    feats = torch.as_tensor(obj).float()
    return feats.unsqueeze(0) if feats.dim() == 1 else feats


class BagSet:
    """Patient-level (column 'paths') or slide-level (column 'path') manifest.
    All bags are loaded once and kept in memory (float16)."""

    def __init__(self, manifest_csv):
        self.df = pd.read_csv(manifest_csv)
        cols = set(self.df.columns)
        if "paths" in cols:
            self.path_lists = [p.split(";") for p in self.df["paths"]]
        elif "path" in cols:
            self.path_lists = [[p] for p in self.df["path"]]
        else:
            raise ValueError(f"{manifest_csv}: needs a 'paths' or 'path' column")
        self.id_col = "patient_id" if "patient_id" in cols else "slide_id"
        if "RSHigh_classic" not in cols or "RS" not in cols:
            raise ValueError(f"{manifest_csv}: needs 'RSHigh_classic' and 'RS' columns")
        labels = self.df["RSHigh_classic"].map({"H": 1, "L": 0})
        if labels.isna().any():
            raise ValueError(f"{manifest_csv}: unexpected RSHigh_classic values "
                             f"{self.df['RSHigh_classic'].unique()}")
        self.labels = labels.astype(int).values
        n_bad = int((self.labels != (self.df["RS"] > RS_THRESHOLD).astype(int).values).sum())
        if n_bad:
            print(f"WARNING {manifest_csv}: {n_bad} rows where RSHigh_classic disagrees with RS > 25")
        missing = [p for ps in self.path_lists for p in ps if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} feature files missing, e.g. {missing[:3]}")
        self.bags = [torch.cat([load_feature_file(p) for p in ps]).half()
                     for ps in tqdm(self.path_lists, desc=f"Loading {os.path.basename(manifest_csv)}")]

    def __len__(self):
        return len(self.bags)


# ----------------------------------------------------------------- model
def create_model_from_config(config, device):
    arch = config['mil_architecture']
    input_dim = INPUT_DIMS[config['model_name']]
    p = config.get('architecture_specific_params', {})
    if arch == "mean":
        model = MeanPoolingMILClassifier(input_dim=input_dim, hidden_dim=p['hidden_dim'],
                                         n_classes=2, dropout=p['dropout'])
    elif arch == "attention":
        model = AttentionMILClassifier(input_dim=input_dim, hidden_dim=p['hidden_dim'],
                                       attention_hidden_dim=p['attention_hidden_dim'],
                                       n_classes=2, dropout=p['dropout'])
    elif arch == "clam":
        model = CLAMClassifier(input_dim=input_dim, hidden_dim=p['hidden_dim'],
                               attention_hidden_dim=p['attention_hidden_dim'], n_classes=2,
                               dropout=p['dropout'], gate=p['gate'],
                               instance_eval=p['use_instance_learning'],
                               k_sample=p['k_sample'], instance_loss_fn=p['instance_loss_fn'])
    elif arch == "acmil":
        model = ACMILClassifier(input_dim=input_dim, hidden_dim=p['hidden_dim'], n_classes=2,
                                n_branches=p['n_branches'], dropout=p['dropout'],
                                top_k=p['top_k'], mask_ratio=p['mask_ratio'],
                                lambda_p=p['lambda_p'], lambda_d=p['lambda_d'], gate=p['gate'])
    else:
        raise ValueError(f"Unknown MIL architecture: {arch}")
    return model.to(device)


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    probs = []
    for bag in data.bags:
        logits = model(bag.float().to(device))
        if logits.dim() == 2 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        probs.append(torch.softmax(logits, dim=0).cpu().numpy())
    return np.array(probs)


# ----------------------------------------------------------------- plots
def plot_all_roc_curves(all_probs, labels, fold_ids, output_dir, title):
    plt.figure(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fold, probs in zip(fold_ids, all_probs):
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        plt.plot(fpr, tpr, linewidth=1.5, alpha=0.4, label=f'Fold {fold} (AUC = {auc(fpr, tpr):.3f})')
        t = np.interp(mean_fpr, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([auc(mean_fpr, t) for t in tprs])
    plt.plot(mean_fpr, mean_tpr, 'b-', linewidth=2.5, alpha=0.8,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                     color='grey', alpha=0.2, label='± 1 std. dev.')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_folds_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'all_folds_roc_curves.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def plot_fold_comparison(fold_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, metric in zip(axes.ravel(), ['auroc', 'sensitivity', 'specificity', 'f1_score']):
        if metric not in fold_df.columns:
            continue
        vals = fold_df[metric].values
        bars = ax.bar(fold_df['fold'].values, vals, color='skyblue', edgecolor='navy')
        bars[int(np.argmax(vals))].set_color('green')
        bars[int(np.argmin(vals))].set_color('red')
        ax.axhline(y=vals.mean(), color='orange', linestyle='--', label=f'Mean: {vals.mean():.3f}')
        ax.set_xlabel('Fold'); ax.set_ylabel(metric.upper()); ax.set_title(f'{metric.upper()} Across Folds')
        ax.set_ylim([0, 1]); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_performance_comparison.png'), dpi=300)
    plt.close()


# ------------------------------------------------------ detailed outputs
def save_detailed_predictions(all_probs, fold_ids, data, output_dir):
    df = data.df
    out = pd.DataFrame({
        data.id_col: df[data.id_col].values,
        'true_label': data.labels,
        'RS_score': df['RS'].values,
    })
    for col in ['n_slides', 'cohort', 'race_group', 'slide_ids']:
        if col in df.columns:
            out[col] = df[col].values
    p1 = np.stack([p[:, 1] for p in all_probs], axis=1)     # samples x folds
    for j, fold in enumerate(fold_ids):
        out[f'fold_{fold}_prob_class1'] = p1[:, j]
        out[f'fold_{fold}_pred'] = (all_probs[j][:, 1] > all_probs[j][:, 0]).astype(int)
    preds = np.stack([out[f'fold_{f}_pred'] for f in fold_ids], axis=1)
    out['ensemble_prob_class1'] = p1.mean(axis=1)
    out['ensemble_prob_std'] = p1.std(axis=1)
    out['ensemble_pred'] = (out['ensemble_prob_class1'] >= 0.5).astype(int)
    out['majority_vote_pred'] = (preds.mean(axis=1) > 0.5).astype(int)
    out['vote_confidence'] = np.maximum(preds.mean(axis=1), 1 - preds.mean(axis=1))
    out['fold_agreement'] = (preds.min(axis=1) == preds.max(axis=1)).astype(int)
    out['ensemble_correct'] = (out['ensemble_pred'] == out['true_label']).astype(int)
    out['majority_vote_correct'] = (out['majority_vote_pred'] == out['true_label']).astype(int)
    out.sort_values(data.id_col).to_csv(os.path.join(output_dir, 'all_predictions_detailed.csv'), index=False)

    summary = {
        'total_samples': len(out),
        'ensemble_accuracy': float(out['ensemble_correct'].mean()),
        'majority_vote_accuracy': float(out['majority_vote_correct'].mean()),
        'agreement_rate': float(out['fold_agreement'].mean()),
    }
    for c in [0, 1]:
        sub = out[out['true_label'] == c]
        summary[f'class_{c}_samples'] = int(len(sub))
        if len(sub):
            summary[f'class_{c}_ensemble_accuracy'] = float(sub['ensemble_correct'].mean())
    out.nlargest(10, 'ensemble_prob_std')[[data.id_col, 'true_label', 'ensemble_pred', 'ensemble_prob_std']] \
        .to_csv(os.path.join(output_dir, 'high_uncertainty_cases.csv'), index=False)
    dis = out[out['fold_agreement'] == 0][[data.id_col, 'true_label', 'ensemble_pred', 'vote_confidence']]
    if len(dis):
        dis.to_csv(os.path.join(output_dir, 'fold_disagreement_cases.csv'), index=False)
    with open(os.path.join(output_dir, 'prediction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


# ------------------------------------------------------------ evaluation
def evaluate_folder(model_folder, cohort, data, output_dir, device):
    with open(os.path.join(model_folder, "config_used.json")) as f:
        config = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    fold_results, all_probs, fold_ids = [], [], []
    for fold in range(1, N_FOLDS + 1):
        ckpt_path = os.path.join(model_folder, f'best_model_fold_{fold}.pt')
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: missing {ckpt_path}")
            continue
        model = create_model_from_config(config, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        probs = predict(model, data, device)
        preds = np.argmax(probs, axis=1)

        ev = ClassificationEvaluator(n_classes=2)
        ev.update(data.labels, preds, probs)
        metrics = ev.compute_all_metrics(verbose=False)
        ev.plot_confusion_matrix(save_path=os.path.join(output_dir, f'fold_{fold}_confusion_matrix.png'),
                                 title=f'Fold {fold} - Confusion Matrix')
        ev.plot_roc_curve(save_path=os.path.join(output_dir, f'fold_{fold}_roc_curve.png'),
                          title=f'Fold {fold} - ROC Curve (AUC={metrics["auroc"]:.3f})')
        fold_results.append({'fold': fold,
                             'val_auroc': ckpt.get('best_auroc', ckpt.get('metrics', {}).get('auroc', 0)),
                             'best_epoch': ckpt.get('epoch', 0), **metrics})
        all_probs.append(probs)
        fold_ids.append(fold)

    if not fold_results:
        print(f"  No fold models found in {model_folder}")
        return None

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)
    pred_summary = save_detailed_predictions(all_probs, fold_ids, data, output_dir)
    title = 'Internal Test - ROC Curves' if cohort == 'internal' else 'External (ECU) - ROC Curves'
    plot_all_roc_curves(all_probs, data.labels, fold_ids, output_dir, title)
    plot_fold_comparison(fold_df, output_dir)

    stats = {m: {'mean': float(fold_df[m].mean()), 'std': float(fold_df[m].std()),
                 'min': float(fold_df[m].min()), 'max': float(fold_df[m].max())}
             for m in METRICS if m in fold_df.columns}
    with open(os.path.join(output_dir, 'individual_results.json'), 'w') as f:
        json.dump({'model_folder': model_folder, 'cohort': cohort,
                   'n_samples': len(data), 'n_high_risk': int(data.labels.sum()),
                   'label_rule': 'RSHigh_classic (H if RS > 25)', 'n_folds': len(fold_results),
                   'fold_metrics': fold_results, 'summary_statistics': stats,
                   'prediction_summary': pred_summary, 'config': config},
                  f, indent=2, cls=NumpyEncoder)

    line = "  ".join(f"{m}={stats[m]['mean']:.3f}±{stats[m]['std']:.3f}"
                     for m in ['auroc', 'auc_pr', 'balanced_accuracy', 'sensitivity', 'specificity']
                     if m in stats)
    print(f"  {cohort:8s} n={len(data)} (HR={int(data.labels.sum())})  {line}")
    return {'model': config['model_name'], 'mil': config['mil_architecture'], 'cohort': cohort,
            'n': len(data), **{f'{m}_mean': v['mean'] for m, v in stats.items()},
            **{f'{m}_std': v['std'] for m, v in stats.items()}}


def latest_folders(results_root):
    """Latest run folder per encoder+MIL in results_root."""
    best = {}
    for d in sorted(glob.glob(os.path.join(results_root, "*"))):
        m = re.match(r"(.+)_(mean|attention|clam|acmil)_(\d{8}_\d{6})$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "config_used.json")):
            best[(m.group(1), m.group(2))] = d       # sorted, so last = latest timestamp
    return [best[k] for k in sorted(best)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Patient-level inference on internal test and ECU")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--model-folder', nargs='+', help='one or more training result folders')
    g.add_argument('--all', action='store_true', help='all configurations in --results-root')
    ap.add_argument('--results-root', default='retrained_models_patient_level')
    ap.add_argument('--cohorts', nargs='+', default=['internal', 'ecu'], choices=['internal', 'ecu'])
    ap.add_argument('--output-root', default='detailed_predictions_patient_level')
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folders = latest_folders(args.results_root) if args.all else args.model_folder
    print(f"Device: {device} | {len(folders)} configuration(s) | cohorts: {args.cohorts}")

    data_cache, cache_enc, summary_rows = {}, None, []
    for folder in folders:
        with open(os.path.join(folder, "config_used.json")) as f:
            cfg = json.load(f)
        enc, mil = cfg['model_name'], cfg['mil_architecture']
        print(f"\n=== {enc} + {mil}  ({folder})")
        if enc != cache_enc:            # new encoder: free the previous encoder's features
            data_cache, cache_enc = {}, enc
        for cohort in args.cohorts:
            manifest = MANIFESTS[cohort].format(enc=enc)
            if manifest not in data_cache:
                data_cache[manifest] = BagSet(manifest)
            out_dir = os.path.join(args.output_root, cohort, f"{enc}_{mil}")
            row = evaluate_folder(folder, cohort, data_cache[manifest], out_dir, device)
            if row:
                summary_rows.append(row)

    if summary_rows:
        os.makedirs(args.output_root, exist_ok=True)
        path = os.path.join(args.output_root, "summary_all_models.csv")
        new = pd.DataFrame(summary_rows)
        if os.path.exists(path):   # update rows for re-run configurations, keep the rest
            old = pd.read_csv(path)
            keys = set(zip(new.model, new.mil, new.cohort))
            old = old[[k not in keys for k in zip(old.model, old.mil, old.cohort)]]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(['cohort', 'model', 'mil']).to_csv(path, index=False)
        print(f"\nSummary table: {path}")