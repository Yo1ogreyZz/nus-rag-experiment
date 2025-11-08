import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) DATA — EDIT HERE IF NEEDED
# ----------------------------
models = ["qwen2.5_3b", "qwen2.5_1.5b", "llama3.2_3b", "deepseek-r1_1.5b", "phi3_mini"]

# Overall scores from your tables
benchmark_overall = [0.786, 0.744, 0.717, 0.645, 0.490]  # Benchmark (origin) Overall_weighted
gpt5_overall      = [0.737, 0.703, 0.718, 0.603, 0.421]  # GPT-5 (mine) Overall_weighted

# Categories for radar
categories = ["factual", "procedural", "comparative", "recommendation"]

# If you have real per-category means, put them here (one list aligned with `models` for each category).
# Defaults below simply copy the overall score to each category so the code runs out-of-the-box.
benchmark_cat = {
    "factual":        benchmark_overall[:],
    "procedural":     benchmark_overall[:],
    "comparative":    benchmark_overall[:],
    "recommendation": benchmark_overall[:],
}
gpt5_cat = {
    "factual":        gpt5_overall[:],
    "procedural":     gpt5_overall[:],
    "comparative":    gpt5_overall[:],
    "recommendation": gpt5_overall[:],
}

# -------------------------
# 2) OUTPUT DIRECTORY SETUP
# -------------------------
out_dir = os.path.join(".", "plot")
os.makedirs(out_dir, exist_ok=True)
saved_paths = []

# --------------------------------------------
# 3) HELPERS: BAR CHART & RADAR CHART DRAWERS
# --------------------------------------------
def save_overall_bar(models, scores, title, outfile):
    fig = plt.figure(figsize=(9, 5))
    x = np.arange(len(models))
    plt.bar(x, scores)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, outfile)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    saved_paths.append(fig_path)

def save_radar(models, categories, per_model_values, title, outfile):
    """
    per_model_values: dict {model_name: [v_cat1, v_cat2, v_cat3, v_cat4]}
    """
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close the loop

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)  # start from top
    ax.set_theta_direction(-1)      # clockwise

    # Category labels around the circle
    ax.set_thetagrids(angles[:-1] * 180/np.pi, categories)
    ax.set_ylim(0.0, 1.0)

    # One polygon per model (default colors)
    for m in models:
        vals = per_model_values[m]
        vals = np.array(vals, dtype=float)
        if vals.size != len(categories):
            raise ValueError(f"Model '{m}' has {vals.size} values; expected {len(categories)}")
        vals = np.concatenate([vals, vals[:1]])  # close polygon
        ax.plot(angles, vals, label=m)
        ax.fill(angles, vals, alpha=0.1)

    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    plt.tight_layout()
    fig_path = os.path.join(out_dir, outfile)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    saved_paths.append(fig_path)

# ---------------------------------------------------
# 4) CONVERT CATEGORY DICTS -> PER-MODEL VALUE TABLES
# ---------------------------------------------------
def to_per_model_dict(models, categories, cat_dict):
    per_model = {}
    for i, m in enumerate(models):
        per_model[m] = [cat_dict[c][i] for c in categories]
    return per_model

benchmark_per_model = to_per_model_dict(models, categories, benchmark_cat)
gpt5_per_model      = to_per_model_dict(models, categories, gpt5_cat)

# -----------------------
# 5) DRAW & SAVE FIGURES
# -----------------------
save_overall_bar(models, benchmark_overall,
                 "Benchmark — Overall Weighted Scores by Model",
                 "benchmark_overall_bar.png")

save_overall_bar(models, gpt5_overall,
                 "GPT-5 — Overall Weighted Scores by Model",
                 "gpt5_overall_bar.png")

save_radar(models, categories, benchmark_per_model,
           "Benchmark — Category Scores (Radar)",
           "benchmark_radar.png")

save_radar(models, categories, gpt5_per_model,
           "GPT-5 — Category Scores (Radar)",
           "gpt5_radar.png")

# -------------------------------------
# 6) SAVE FILENAMES (TXT + JSON MANIFEST)
# -------------------------------------
txt_path = os.path.join(out_dir, "figures_list.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for p in saved_paths:
        f.write(p + "\n")

json_path = os.path.join(out_dir, "figures_list.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(saved_paths, f, ensure_ascii=False, indent=2)