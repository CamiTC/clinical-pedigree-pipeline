import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = "outputs/graphs"

task_labels = [
    "Family\nMembers", "Conditions", "Relative–\nCondition",
    "Side", "Polarity", "Deceased", "Sex", "Age at\nEvent", "Full\nRecord"
]

task_keys = [
    "family_members", "conditions", "relative_condition",
    "side", "polarity", "deceased", "sex", "age_at_event", "full_record"
]

# load all per-transcript evaluation CSVs
per_transcript_dir = "outputs/evaluation/per_transcript"
dfs = [pd.read_csv(file) for file in sorted(Path(per_transcript_dir).glob("*_evaluation.csv"))]
df = pd.concat(dfs, ignore_index=True)

nlp_df = df[df["method"] == "NLP"]
llm_df = df[df["method"] == "NLP + LLM"]

nlp_mean = [nlp_df[nlp_df["task"] == task]["F1"].mean() for task in task_keys]
nlp_std = [nlp_df[nlp_df["task"] == task]["F1"].std() for task in task_keys]
nlp_min = [nlp_df[nlp_df["task"] == task]["F1"].min() for task in task_keys]
nlp_max = [nlp_df[nlp_df["task"] == task]["F1"].max() for task in task_keys]

llm_mean = [llm_df[llm_df["task"] == task]["F1"].mean() for task in task_keys]
llm_std = [llm_df[llm_df["task"] == task]["F1"].std() for task in task_keys]
llm_min = [llm_df[llm_df["task"] == task]["F1"].min() for task in task_keys]
llm_max = [llm_df[llm_df["task"] == task]["F1"].max() for task in task_keys]

x = np.arange(len(task_labels))
width = 0.35

# --------------------
# average F1 bar chart
# --------------------

fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - width/2, nlp_mean, width, label="NLP", color="#5B9BD5", alpha=0.85, zorder=3)
bars2 = ax.bar(x + width/2, llm_mean, width, label="NLP-LLM", color="#ED7D31", alpha=0.85, zorder=3)

ax.errorbar(x - width/2, nlp_mean, yerr=nlp_std, fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
ax.errorbar(x + width/2, llm_mean, yerr=llm_std, fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

for i in range(len(task_labels)):
    ax.plot(x[i] - width/2, nlp_min[i], "v", color="#2E6FA3", markersize=5, zorder=5)
    ax.plot(x[i] - width/2, nlp_max[i], "^", color="#2E6FA3", markersize=5, zorder=5)
    ax.plot(x[i] + width/2, llm_min[i], "v", color="#A84E00", markersize=5, zorder=5)
    ax.plot(x[i] + width/2, llm_max[i], "^", color="#A84E00", markersize=5, zorder=5)

for i, (m, s) in enumerate(zip(nlp_mean, nlp_std)):
    ax.text(x[i] - width/2 - 0.02, m + 0.05, f"±{s:.2f}", ha="right", va="center", fontsize=6.5, color="#2E6FA3")
for i, (m, s) in enumerate(zip(llm_mean, llm_std)):
    ax.text(x[i] + width/2 + 0.02, m + 0.05, f"±{s:.2f}", ha="left", va="center", fontsize=6.5, color="#A84E00")

ax.set_ylabel("F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend(handles=[
    bars1, bars2,
    mlines.Line2D([], [], color="gray", marker="^", linestyle="None", markersize=5, label="Max"),
    mlines.Line2D([], [], color="gray", marker="v", linestyle="None", markersize=5, label="Min"),
])
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/f1_grouped_bar.png", dpi=150)
print("Saved f1_grouped_bar.png")
plt.close()

# -----------
# F1 Box Plot
# -----------

nlp_data = [nlp_df[nlp_df["task"] == t]["F1"].values for t in task_keys]
llm_data = [llm_df[llm_df["task"] == t]["F1"].values for t in task_keys]

fig, ax = plt.subplots(figsize=(13, 6))

bp1 = ax.boxplot(nlp_data, positions=x - width/2, widths=width * 0.9,
                 patch_artist=True,
                 boxprops=dict(facecolor="#5B9BD5", alpha=0.85),
                 medianprops=dict(color="black", linewidth=1.5),
                 whiskerprops=dict(color="#2E6FA3"),
                 capprops=dict(color="#2E6FA3"),
                 flierprops=dict(marker="o", color="#2E6FA3", markersize=4, alpha=0.6))

bp2 = ax.boxplot(llm_data, positions=x + width/2, widths=width * 0.9,
                 patch_artist=True,
                 boxprops=dict(facecolor="#ED7D31", alpha=0.85),
                 medianprops=dict(color="black", linewidth=1.5),
                 whiskerprops=dict(color="#A84E00"),
                 capprops=dict(color="#A84E00"),
                 flierprops=dict(marker="o", color="#A84E00", markersize=4, alpha=0.6))

ax.set_ylabel("F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=9)
ax.set_ylim(0, 1.15)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(handles=[
    mpatches.Patch(facecolor="#5B9BD5", alpha=0.85, label="NLP"),
    mpatches.Patch(facecolor="#ED7D31", alpha=0.85, label="NLP-LLM"),
])
plt.tight_layout()
plt.savefig(f"{OUT}/f1_boxplot.png", dpi=300, transparent=True)
plt.close()

# -------------------------------------
# synthetic transcript scores bar chart
# -------------------------------------

categories = ["Coherency", "Consistency", "Fluency", "Relevance"]
raw = {
    "Coherency": [4, 4, 3, 4, 4, 4, 4, 3, 4, 3],
    "Consistency": [4, 4, 3, 4, 4, 3, 3, 3, 3, 2],
    "Fluency": [4, 4, 3, 4, 4, 4, 4, 3, 3, 3],
    "Relevance": [4, 4, 3, 4, 4, 4, 4, 3, 4, 3],
}
colors = ["#5B9BD5", "#ED7D31", "#A9D18E", "#FFC000"]
means = [np.mean(raw[c]) for c in categories]
stds = [np.std(raw[c]) for c in categories]
xc = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(xc, means, width=0.5, color=colors, alpha=0.8, zorder=3)
ax.errorbar(xc, means, yerr=stds, fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4)

for i, cat in enumerate(categories):
    jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=10)
    ax.scatter(xc[i] + jitter, raw[cat], color="black", s=25, zorder=5, alpha=0.6)

for bar, mean, std in zip(bars, means, stds):
    cx = bar.get_x() + bar.get_width() / 2
    cy = bar.get_height() / 2 + 0.8
    ax.text(cx, cy + 0.12, f"{mean:.2f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(cx, cy - 0.18, f"±{std:.2f}", ha="center", va="center", fontsize=8, color="white")

ax.set_xticks(xc)
ax.set_xticklabels(categories)
ax.set_ylabel("Score")
ax.set_ylim(1, 5.3)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/synthetic_bar.png", dpi=150)
plt.close()


# ------------------------------------
# synthetic transcript scores box plot
# ------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([raw[c] for c in categories], patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(marker="o", markersize=5, alpha=0.6))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

for i, cat in enumerate(categories):
    jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=len(raw[cat]))
    ax.scatter(np.ones(len(raw[cat])) * (i + 1) + jitter, raw[cat], color="black", s=25, zorder=5, alpha=0.6)

ax.set_xticks(range(1, len(categories) + 1))
ax.set_xticklabels(categories)
ax.set_ylabel("Score")
ax.set_ylim(1, 5.3)
ax.set_yticks([1, 2, 3, 4, 5])
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/synthetic_boxplot.png", dpi=150)
plt.close()


# -----------------------------
# individual F1 scores heat map
# -----------------------------

transcript_ids = []
nlp_full = []
llm_full = []

for file in sorted(Path(per_transcript_dir).glob("*_evaluation.csv")):
    transcript_id = file.stem.replace("_evaluation", "")
    transcript_ids.append(transcript_id.upper())
    transcript_df = pd.read_csv(file)
    nlp_val = transcript_df[(transcript_df["method"] == "NLP") & (transcript_df["task"] == "full_record")]["F1"].values
    llm_val = transcript_df[(transcript_df["method"] == "NLP + LLM") & (transcript_df["task"] == "full_record")]["F1"].values
    nlp_full.append(nlp_val[0] if len(nlp_val) else 0)
    llm_full.append(llm_val[0] if len(llm_val) else 0)

data = np.array([nlp_full, llm_full])

fig, ax = plt.subplots(figsize=(11, 2.8))
im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

ax.set_xticks(range(len(transcript_ids)))
ax.set_xticklabels(transcript_ids)
ax.set_yticks([0, 1])
ax.set_yticklabels(["NLP", "NLP-LLM"])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        colour = "black" if 0.3 < val < 0.75 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=colour, fontweight="bold")

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="F1")
plt.tight_layout()
plt.savefig(f"{OUT}/full_record_heatmap.png", dpi=150)
plt.close()
