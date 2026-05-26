from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

import model


def _infer_sample_freq(R: int) -> str:
    if R == 5:
        return "week"
    if R == 20:
        return "month"
    if R in {60, 65}:
        return "quarter"
    raise ValueError("Please provide --dataset-dir explicitly when R is not 5, 20, 60, or 65.")


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _format_float(value: float) -> str:
    text = f"{value:.0e}" if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e3) else f"{value:.6f}"
    return text.replace("+", "").replace(".", "p")


def _build_run_name(config: dict) -> str:
    parts = [
        f"I{config['I']}",
        f"R{config['R']}",
        f"bs{config['batch_size']}",
        f"ep{config['epochs']}",
        f"lr{_format_float(config['lr'])}",
        f"wd{_format_float(config['weight_decay'])}",
        f"vr{str(config['val_ratio']).replace('.', 'p')}",
        f"es{config['early_stop_patience']}",
        f"ens{len(config['seeds'])}",
    ]
    if config.get("tag"):
        parts.append(str(config["tag"]))
    return "_".join(parts)


def _load_meta(dataset_dir: Path, stem: str):
    meta_csv = dataset_dir / f"{stem}.csv"
    meta_feather = dataset_dir / f"{stem}.feather"
    if meta_csv.exists():
        meta = pd.read_csv(meta_csv)
    elif meta_feather.exists():
        meta = pd.read_feather(meta_feather)
    else:
        return None

    for col in ["start_date", "date", "label_end_date"]:
        if col in meta.columns:
            meta[col] = pd.to_datetime(meta[col], errors="coerce")
    return meta


def _meta_years(meta: pd.DataFrame | None):
    if meta is None or "date" not in meta.columns:
        return None
    return pd.to_datetime(meta["date"], errors="coerce").dt.year.to_numpy()


def _load_dataset(dataset_dir: Path):
    X_trainval = np.load(dataset_dir / "X_trainval.npy")
    y_trainval = np.load(dataset_dir / "y_trainval.npy")
    X_test = np.load(dataset_dir / "X_test.npy")
    y_test = np.load(dataset_dir / "y_test.npy")

    meta_trainval = _load_meta(dataset_dir, "meta_trainval")
    meta_test = _load_meta(dataset_dir, "meta_test")

    return X_trainval, y_trainval, X_test, y_test, meta_trainval, meta_test


def _save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _evaluate_ensemble(X_test, y_test, checkpoint_paths, batch_size, device, years_test=None):
    pred_prob = model.average_ensemble_predictions(
        X_test,
        checkpoint_paths,
        batch_size=batch_size,
        device=device,
        years=years_test,
    )
    pred_prob = np.asarray(pred_prob).reshape(-1)
    pred_label = (pred_prob > 0.5).astype(int)

    metrics = {
        "test_mse": float(mean_squared_error(y_test, pred_prob)),
        "test_acc": float(accuracy_score(y_test, pred_label)),
        "test_logloss": float(log_loss(y_test, pred_prob)),
        "test_positive_rate": float(pred_label.mean()),
        "test_prob_mean": float(pred_prob.mean()),
        "test_prob_std": float(pred_prob.std()),
    }
    return pred_prob, pred_label, metrics


def run_one_config(
    *,
    dataset_dir: Path,
    output_root: Path,
    Market: str,
    I: int,
    R: int,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_ratio: float,
    early_stop_patience: int,
    num_workers: int,
    device: str,
    tag: str | None = None,
):
    config = {
        "Market": Market,
        "dataset_dir": str(dataset_dir),
        "I": I,
        "R": R,
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "early_stop_patience": early_stop_patience,
        "num_workers": num_workers,
        "device": device,
        "tag": tag,
    }
    run_name = _build_run_name(config)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    X_trainval, y_trainval, X_test, y_test, meta_trainval, meta_test = _load_dataset(dataset_dir)
    _save_json(run_dir / "config.json", config)
    years_trainval = _meta_years(meta_trainval)
    years_test = _meta_years(meta_test)

    results = model.train_resplit_ensemble(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        I=I,
        R=R,
        seeds=tuple(seeds),
        years_trainval=years_trainval,
        save_dir=str(run_dir),
        Market=Market,
        val_ratio=val_ratio,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        device=device,
        early_stop_patience=early_stop_patience,
    )

    per_seed_rows = []
    history_frames = []
    checkpoint_paths = []
    for res in results:
        checkpoint_paths.append(res["best_path"])
        per_seed_rows.append(
            {
                "seed": res["seed"],
                "best_path": res["best_path"],
                "history_path": res["history_path"],
                "best_score": res["best_score"],
                "best_val_loss": res["best_val_loss"],
                "best_epoch": res["best_epoch"],
                "stopped_early": res["stopped_early"],
                "epochs_ran": len(res["history"]),
            }
        )
        hist = pd.DataFrame(res["history"]).copy()
        hist["run_name"] = run_name
        history_frames.append(hist)

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)
    per_seed_df.to_csv(run_dir / "per_seed_summary.csv", index=False, encoding="utf-8-sig")

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    history_df.to_csv(run_dir / "all_history.csv", index=False, encoding="utf-8-sig")

    pred_prob, pred_label, test_metrics = _evaluate_ensemble(
        X_test=X_test,
        y_test=y_test,
        checkpoint_paths=checkpoint_paths,
        batch_size=batch_size,
        device=device,
        years_test=years_test,
    )

    prediction_df = pd.DataFrame(
        {
            "row_id": np.arange(len(y_test)),
            "y_true": np.asarray(y_test, dtype=np.int64),
            "pred_prob": pred_prob,
            "pred_label": pred_label,
        }
    )
    if meta_test is not None:
        prediction_df = pd.concat([meta_test.reset_index(drop=True), prediction_df], axis=1)
    prediction_df.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "run_name": run_name,
        "Market": Market,
        "I": I,
        "R": R,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "early_stop_patience": early_stop_patience,
        "num_workers": num_workers,
        "device": device,
        "n_seeds": len(seeds),
        "test_n": int(len(y_test)),
        "mean_best_val_loss": float(per_seed_df["best_val_loss"].mean()),
        "mean_best_epoch": float(per_seed_df["best_epoch"].mean()),
        "n_stopped_early": int(per_seed_df["stopped_early"].sum()),
        **test_metrics,
    }
    pd.DataFrame([run_summary]).to_csv(run_dir / "run_summary.csv", index=False, encoding="utf-8-sig")
    return run_summary


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps and save train/test results.")
    parser.add_argument("--market", default="US_1993_2020")
    parser.add_argument("--I", type=int, required=True)
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-root", default="models/sweeps")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", default="50")
    parser.add_argument("--batch-sizes", default="128")
    parser.add_argument("--lrs", default="1e-4")
    parser.add_argument("--weight-decays", default="1e-4")
    parser.add_argument("--val-ratios", default="0.3")
    parser.add_argument("--early-stop-patiences", default="5")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    sample_freq = _infer_sample_freq(args.R)
    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else Path("data/training_data") / args.market / f"I{args.I}R{args.R}S{args.R}_period_end_{sample_freq}"
    )
    output_root = Path(args.output_root) / args.market / f"I{args.I}R{args.R}"
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_int_list(args.seeds)
    epochs_list = _parse_int_list(args.epochs)
    batch_sizes = _parse_int_list(args.batch_sizes)
    lrs = _parse_float_list(args.lrs)
    weight_decays = _parse_float_list(args.weight_decays)
    val_ratios = _parse_float_list(args.val_ratios)
    early_stop_patiences = _parse_int_list(args.early_stop_patiences)

    all_run_summaries = []
    grid = itertools.product(
        epochs_list,
        batch_sizes,
        lrs,
        weight_decays,
        val_ratios,
        early_stop_patiences,
    )
    for epochs, batch_size, lr, weight_decay, val_ratio, early_stop_patience in grid:
        summary = run_one_config(
            dataset_dir=dataset_dir,
            output_root=output_root,
            Market=args.market,
            I=args.I,
            R=args.R,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            val_ratio=val_ratio,
            early_stop_patience=early_stop_patience,
            num_workers=args.num_workers,
            device=args.device,
            tag=args.tag or None,
        )
        all_run_summaries.append(summary)

    summary_df = pd.DataFrame(all_run_summaries).sort_values(
        ["mean_best_val_loss", "test_mse", "test_acc"],
        ascending=[True, True, False],
    )
    summary_df.to_csv(output_root / "sweep_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df)


if __name__ == "__main__":
    main()
