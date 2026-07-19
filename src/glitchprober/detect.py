"""GlitchProber detection (paper Algorithm 1).

sample gamma of vocab -> label via repetition task -> extract key-layer features
-> PCA(P) -> SVM(poly) -> classify the rest -> post-validate every predicted
glitch with the repetition task -> output G, N + metrics vs. RQ1 ground truth.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from ..common.io_utils import Timer
from ..ground_truth.sweep import repetition_sweep
from .features import extract_features


def run_detection(
    model,
    tok,
    mcfg: dict,
    candidates: list[int],
    true_glitch: set[int],
    det_cfg: dict,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    checkpoint_dir=None,
    task: str = "repetition",
    correct_fn=None,
) -> dict:
    rng = np.random.default_rng(seed)
    gamma = det_cfg["gamma"]
    features = det_cfg["features"]
    ckpt = (lambda name: checkpoint_dir / f"{name}_seed{seed}.csv") if checkpoint_dir else (lambda name: None)

    with Timer() as t_total:
        # --- sample + label (training data) ---
        n_sample = max(2, int(len(candidates) * gamma))
        sample = list(rng.choice(candidates, size=n_sample, replace=False))
        sample_results = repetition_sweep(
            model, tok, sample, batch_size, max_new_tokens, desc="label sample",
            checkpoint=ckpt("label_sample"), task=task, correct_fn=correct_fn,
        )
        y = np.array([0 if sample_results[t][0] else 1 for t in sample])  # 1 = glitch

        # --- features + PCA + SVM ---
        X = extract_features(model, tok, sample, mcfg, features, batch_size, "sample feats", task).astype(np.float32)
        pca = PCA(n_components=min(det_cfg["pca_dim"], len(sample) - 1), random_state=seed)
        Xp = pca.fit_transform(X)
        svm = SVC(
            kernel=det_cfg["svm"]["kernel"],
            C=det_cfg["svm"]["C"],
            degree=det_cfg["svm"]["degree"],
            random_state=seed,
        )
        svm.fit(Xp, y)

        # --- classify the unsampled rest, chunked to bound RAM; chunk-level checkpoint ---
        sample_set = set(sample)
        rest = [t for t in candidates if t not in sample_set]
        pred_done: dict[int, int] = {}
        pred_ckpt = ckpt("svm_predictions")
        if pred_ckpt and pred_ckpt.exists():
            prev = pd.read_csv(pred_ckpt)
            pred_done = dict(zip(prev["token_id"], prev["pred"]))
            print(f"detect: resumed {len(pred_done)} SVM predictions from checkpoint")
        todo = [t for t in rest if t not in pred_done]
        chunk_size = 2048
        for i in range(0, len(todo), chunk_size):
            chunk = todo[i : i + chunk_size]
            Xc = extract_features(model, tok, chunk, mcfg, features, batch_size, "classify", task).astype(np.float32)
            pred = svm.predict(pca.transform(Xc))
            for t, p in zip(chunk, pred):
                pred_done[t] = int(p)
            if pred_ckpt:
                pd.DataFrame({"token_id": list(pred_done), "pred": list(pred_done.values())}).to_csv(pred_ckpt, index=False)
        predicted_glitch = [t for t in rest if pred_done.get(t) == 1]

        # --- post-validation of positives (paper Sec. 4.1.3) ---
        G = set(t for t in sample if not sample_results[t][0])
        if det_cfg.get("post_validation", True):
            val = repetition_sweep(
                model, tok, predicted_glitch, batch_size, max_new_tokens,
                desc="post-validate", checkpoint=ckpt("post_validate"),
                task=task, correct_fn=correct_fn,
            )
            G |= {t for t, (ok, _) in val.items() if not ok}
        else:
            G |= set(predicted_glitch)

    # --- metrics vs. ground truth ---
    tp = len(G & true_glitch)
    fp = len(G - true_glitch)
    fn = len(true_glitch - G)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "seed": seed,
        "n_candidates": len(candidates),
        "n_sampled": n_sample,
        "n_predicted_glitch_by_svm": len(predicted_glitch),
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "time_seconds": t_total.seconds,
        "paper_claims": {"precision": 1.0, "recall_avg": 0.6447, "f1_avg": 0.7835},
    }
