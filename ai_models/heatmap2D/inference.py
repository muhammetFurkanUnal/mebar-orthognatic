import json
import random
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import cv2

from config import cfg
from dataset import Heatmap2D_Dataset
from model import Heatmap2D_Model
from transforms import test_transform


class InferenceClient:
    def __init__(self, model_class, pth_path, device="cuda"):
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model_class()
        checkpoint = torch.load(
            pth_path, map_location=self.device, weights_only=True
        )
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.inference_mode():
            infered = self.model(image)

        return infered[0].detach().cpu()


def normalize_img_to_RGB(image):
    image = image.detach().cpu().float()
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def get_peak_coords(heatmap):
    flat_idx = torch.argmax(heatmap).item()
    w = heatmap.shape[-1]
    y = flat_idx // w
    x = flat_idx % w
    return int(x), int(y)


def infer(
    dataset_ins: Heatmap2D_Dataset,
    model_ins,
    pth_path,
    sample=None,
    image_path=None,
    output_path="inference.png",
    device="cuda",
):
    client = InferenceClient(
        model_class=model_ins, pth_path=pth_path, device=device
    )

    if sample is None:
        if image_path is not None:
            found = False
            for s in dataset_ins:
                if str(getattr(s, "img_path", "")) == str(image_path):
                    sample = s
                    found = True
                    break
            if not found:
                raise ValueError(f"Image path not found in dataset: {image_path}")
        else:
            n = random.randint(0, len(dataset_ins) - 1)
            sample = dataset_ins[n]

    pred_heatmaps = client.infer(sample.image)

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.15, wspace=0.1)

    ax_main = fig.add_subplot(gs[0:2, 1:3])
    img_display = normalize_img_to_RGB(sample.image).permute(1, 2, 0).numpy()
    ax_main.imshow(img_display)

    img_title = Path(sample.img_path).name if hasattr(sample, "img_path") else "Input Image"
    ax_main.set_title(f"Input: {img_title}", fontsize=13, fontweight="bold")
    ax_main.axis("off")

    for idx, heatmap in enumerate(sample.heatmaps):
        if idx >= 4:
            break
        ax = fig.add_subplot(gs[2, idx])
        x_tgt, y_tgt = get_peak_coords(heatmap)
        heatmap_norm = normalize_img_to_RGB(heatmap).numpy()
        ax.imshow(heatmap_norm, cmap="magma")
        ax.set_title(f"Target HM {idx}\n(x: {x_tgt}, y: {y_tgt})", fontsize=10)
        ax.axis("off")

    for idx, pred_heatmap in enumerate(pred_heatmaps):
        if idx >= 4:
            break
        ax = fig.add_subplot(gs[3, idx])
        x_pred, y_pred = get_peak_coords(pred_heatmap)
        pred_norm = normalize_img_to_RGB(pred_heatmap).numpy()
        ax.imshow(pred_norm, cmap="magma")
        ax.set_title(f"Pred HM {idx}\n(x: {x_pred}, y: {y_pred})", fontsize=10)
        ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def evaluate_dataset_errors(
    dataset_ins: Heatmap2D_Dataset,
    model_class,
    pth_path,
    output_dir="ai_models/heatmap2D/outputs/val",
    worst_k=5,
    device="cuda",
):
    run_output_dir = Path(output_dir)
    worst_cases_dir = run_output_dir / "worst_cases"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    worst_cases_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Evaluation Output Directory: {run_output_dir}\n")

    client = InferenceClient(
        model_class=model_class, pth_path=pth_path, device=device
    )

    all_errors = []
    sample_records = []

    progress_bar = tqdm(
        range(len(dataset_ins)),
        desc="🔍 Evaluating Dataset",
        unit="sample",
        dynamic_ncols=True,
    )

    for i in progress_bar:
        sample = dataset_ins[i]
        pred_heatmaps = client.infer(sample.image)
        tgt_heatmaps = sample.heatmaps

        num_channels = min(pred_heatmaps.shape[0], tgt_heatmaps.shape[0])
        sample_channel_errors = []

        for c in range(num_channels):
            x_pred, y_pred = get_peak_coords(pred_heatmaps[c])
            x_tgt, y_tgt = get_peak_coords(tgt_heatmaps[c])

            l2_error = float(np.sqrt((x_pred - x_tgt) ** 2 + (y_pred - y_tgt) ** 2))
            all_errors.append(l2_error)
            sample_channel_errors.append(l2_error)

        max_sample_error = float(max(sample_channel_errors))
        mean_sample_error = float(np.mean(sample_channel_errors))
        img_path = str(getattr(sample, "img_path", f"sample_{i}"))

        sample_records.append({
            "index": i,
            "sample": sample,
            "img_path": img_path,
            "max_error": max_sample_error,
            "mean_error": mean_sample_error,
            "channel_errors": sample_channel_errors,
        })

        running_mean = np.mean(all_errors)
        progress_bar.set_postfix({"Running Mean": f"{running_mean:.2f}px"})

    all_errors = np.array(all_errors)

    mean_err = float(np.mean(all_errors))
    median_err = float(np.median(all_errors))
    max_err = float(np.max(all_errors))
    min_err = float(np.min(all_errors))

    percentile_steps = list(range(10, 100, 10))
    percentile_values = np.percentile(all_errors, percentile_steps)
    percentiles_dict = {f"p{p}": float(val) for p, val in zip(percentile_steps, percentile_values)}

    print(f"\n📊 Total Evaluated Keypoints: {len(all_errors)}")
    print(f"🔹 Min Error: {min_err:.4f} px")
    print(f"🔹 Mean Error: {mean_err:.4f} px")
    print(f"🔹 Median Error: {median_err:.4f} px")
    print(f"🔹 Max Error: {max_err:.4f} px")
    print("\n📈 --- Percentile Breakdown ---")
    for p, val in zip(percentile_steps, percentile_values):
        print(f"   P{p:<2}: {val:.4f} px")
    print("--------------------------------\n")

    worst_records = sorted(sample_records, key=lambda x: x["max_error"], reverse=True)[:worst_k]
    
    print(f"🚨 Saving Top {worst_k} Worst Cases...")
    worst_cases_metadata = []
    for rank, item in enumerate(tqdm(worst_records, desc="🖼️ Saving Outliers", unit="img"), start=1):
        file_name = Path(item["img_path"]).name
        save_path = worst_cases_dir / f"worst_{rank}_err_{item['max_error']:.2f}_{file_name}.png"

        infer(
            dataset_ins=dataset_ins,
            model_ins=model_class,
            pth_path=pth_path,
            sample=item["sample"],
            output_path=str(save_path),
            device=device,
        )

        worst_cases_metadata.append({
            "rank": rank,
            "img_path": item["img_path"],
            "max_error": item["max_error"],
            "mean_error": item["mean_error"],
            "channel_errors": item["channel_errors"],
            "visual_output": str(save_path.name),
        })

    plot_path = run_output_dir / "error_distribution.png"
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(all_errors, bins=150, edgecolor="black", alpha=0.6, color="royalblue", label="Errors")
    ax.axvline(mean_err, color="crimson", linestyle="--", linewidth=2.5, label=f"Mean: {mean_err:.2f} px")
    ax.axvline(percentiles_dict["p50"], color="darkorange", linestyle="-", linewidth=2.5, label=f"Median (P50): {percentiles_dict['p50']:.2f} px")

    cmap = plt.colormaps["viridis"].resampled(len(percentile_steps))
    for idx, (p, val) in enumerate(zip(percentile_steps, percentile_values)):
        if p == 50:
            continue
        ax.axvline(val, color=cmap(idx), linestyle=":", linewidth=1.5, alpha=0.85, label=f"P{p}: {val:.2f} px")

    ax.set_title("Coordinate Error Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Euclidean Distance Error (pixels)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(pth_path),
        "total_samples": len(dataset_ins),
        "total_evaluated_keypoints": len(all_errors),
        "metrics": {
            "min_error": min_err,
            "mean_error": mean_err,
            "median_error": median_err,
            "max_error": max_err,
            "percentiles": percentiles_dict,
        },
        "worst_cases": worst_cases_metadata,
    }

    report_path = run_output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Run completed successfully.")
    print(f"📄 Report JSON: {report_path}")
    print(f"📊 Distribution Plot: {plot_path}")

    return report



def darkpose(heatmap, kernel_size=3):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().float().numpy()

    # Ensure C-contiguous float32 numpy array for OpenCV
    heatmap = np.ascontiguousarray(heatmap, dtype=np.float32)

    # 1. Gaussian blur
    heatmap_blurred = cv2.GaussianBlur(
        heatmap, (kernel_size, kernel_size), sigmaX=0
    )

    # 2. Maximum discrete location
    y, x = np.unravel_index(np.argmax(heatmap_blurred), heatmap_blurred.shape)

    # Boundary check for 3x3 window
    h, w = heatmap_blurred.shape
    if 1 <= x < w - 1 and 1 <= y < h - 1:
        val = np.maximum(heatmap_blurred[y - 1 : y + 2, x - 1 : x + 2], 1e-6)
        log_val = np.log(val)

        # 3. Derivatives
        dx = (log_val[1, 2] - log_val[1, 0]) / 2.0
        dy = (log_val[2, 1] - log_val[0, 1]) / 2.0

        dxx = log_val[1, 2] - 2 * log_val[1, 1] + log_val[1, 0]
        dyy = log_val[2, 1] - 2 * log_val[1, 1] + log_val[0, 1]
        dxy = (
            log_val[2, 2] - log_val[2, 0] - log_val[0, 2] + log_val[0, 0]
        ) / 4.0

        # Hessian matrix and gradient vector
        hessian = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float32)
        grad = np.array([dx, dy], dtype=np.float32)

        try:
            delta = -np.linalg.inv(hessian).dot(grad)
        except np.linalg.LinAlgError:
            delta = np.zeros(2, dtype=np.float32)

        subpixel_x = x + delta[0]
        subpixel_y = y + delta[1]
        return float(subpixel_x), float(subpixel_y)

    return float(x), float(y)


def infer_all_on_image(
    dataset_ins: Heatmap2D_Dataset,
    model_class,
    pth_path,
    output_dir="ai_models/heatmap2D/outputs/predictions",
    device="cuda",
    use_darkpose=True,
    draw_ground_truth=True,
    point_radius=1,
):
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(
        model_class=model_class, pth_path=pth_path, device=device
    )

    progress_bar = tqdm(
        range(len(dataset_ins)),
        desc="🎨 Generating Overlaid Predictions",
        unit="img",
        dynamic_ncols=True,
    )

    for i in progress_bar:
        sample = dataset_ins[i]
        pred_heatmaps = client.infer(sample.image)
        tgt_heatmaps = getattr(sample, "heatmaps", None)

        img_tensor = normalize_img_to_RGB(sample.image)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_h, img_w = img_bgr.shape[:2]
        hm_h, hm_w = pred_heatmaps.shape[-2:]
        scale_x = img_w / float(hm_w)
        scale_y = img_h / float(hm_h)

        num_channels = pred_heatmaps.shape[0]

        if draw_ground_truth and tgt_heatmaps is not None:
            num_tgt = min(num_channels, tgt_heatmaps.shape[0])
            for c in range(num_tgt):
                x_gt_hm, y_gt_hm = get_peak_coords(tgt_heatmaps[c])
                gt_x = int(round(x_gt_hm * scale_x))
                gt_y = int(round(y_gt_hm * scale_y))

                cv2.circle(
                    img_bgr,
                    (gt_x, gt_y),
                    point_radius + 1,
                    (0, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )

        for c in range(num_channels):
            if use_darkpose:
                sub_x, sub_y = darkpose(pred_heatmaps[c])
            else:
                sub_x, sub_y = get_peak_coords(pred_heatmaps[c])

            px_x = int(round(sub_x * scale_x))
            px_y = int(round(sub_y * scale_y))

            cv2.circle(
                img_bgr,
                (px_x, px_y),
                point_radius,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

            cv2.putText(
                img_bgr,
                str(c),
                (px_x + 4, px_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if hasattr(sample, "img_path") and sample.img_path:
            out_name = f"{Path(sample.img_path).stem}_pred.png"
        else:
            out_name = f"sample_{i:05d}_pred.png"

        cv2.imwrite(str(save_dir / out_name), img_bgr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Heatmap2D Model Evaluation and Batch Inference Tool"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["eval", "infer"],
        default="eval",
        help="Execution mode: 'eval' runs error distribution analysis, 'infer' draws keypoints on full images",
    )
    parser.add_argument(
        "--worst-k",
        "-k",
        type=int,
        default=5,
        help="Number of worst error samples to visualize and save (eval mode only)",
    )
    parser.add_argument(
        "--use-darkpose",
        action="store_true",
        default=True,
        help="Apply DarkPose subpixel adjustment during inference",
    )
    args = parser.parse_args()

    import glob
    import shutil

    MANIFEST_PATH = "/home/furkan/projects/mebar-orthognatic/data/processed/manifest.csv"
    PROJECT_ROOT = "/home/furkan/projects/mebar-orthognatic"
    CONFIG_PATH = "/home/furkan/projects/mebar-orthognatic/ai_models/heatmap2D/config.json"

    # Version-based output layout: outputs/<version>/{checkpoints,val,predictions}
    VERSION = cfg.version
    VERSION_DIR = Path(PROJECT_ROOT) / "ai_models/heatmap2D/outputs" / VERSION
    CHECKPOINTS_DIR = VERSION_DIR / "checkpoints"
    EVAL_OUTPUT_DIRECTORY = VERSION_DIR / "val"
    INFER_OUTPUT_DIRECTORY = VERSION_DIR / "predictions"

    VERSION_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot the config used for this version (copy only if it does not exist yet)
    version_config_path = VERSION_DIR / "config.json"
    if not version_config_path.exists():
        shutil.copy2(CONFIG_PATH, version_config_path)
        print(f"🧾 Config snapshot saved: {version_config_path}")

    # Auto-discover the best checkpoint for this version
    checkpoint_candidates = sorted(glob.glob(str(CHECKPOINTS_DIR / "best_ep*.pth")))
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No 'best_ep*.pth' checkpoint found for version '{VERSION}' in: {CHECKPOINTS_DIR}"
        )
    CHECKPOINT_PATH = checkpoint_candidates[-1]
    print(f"📦 Using checkpoint: {CHECKPOINT_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Heatmap2D_Dataset(
        df=pd.read_csv(MANIFEST_PATH),
        root_path=PROJECT_ROOT,
        transform=test_transform,
    )

    if args.mode == "eval":
        evaluate_dataset_errors(
            dataset_ins=dataset,
            model_class=Heatmap2D_Model,
            pth_path=CHECKPOINT_PATH,
            output_dir=str(EVAL_OUTPUT_DIRECTORY),
            worst_k=args.worst_k,
            device=device,
        )
    elif args.mode == "infer":
        infer_all_on_image(
            dataset_ins=dataset,
            model_class=Heatmap2D_Model,
            pth_path=CHECKPOINT_PATH,
            output_dir=str(INFER_OUTPUT_DIRECTORY),
            device=device,
            use_darkpose=args.use_darkpose,
            draw_ground_truth=True,
        )