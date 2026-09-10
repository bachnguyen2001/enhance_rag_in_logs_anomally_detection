"""Biểu đồ EDA từ các bảng thống kê; không fit model hoặc chọn preprocessing."""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hdfs-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_eda_plots(output: Path) -> list[Path]:
    frame = pd.read_csv(output / "trace_features.csv")
    events = pd.read_csv(output / "event_statistics.csv").set_index("EventId")
    patterns = pd.read_csv(output / "sequence_patterns.csv", usecols=["blocks"])
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    distribution = frame.Label.value_counts().reindex(["Normal", "Anomaly"])
    distribution.plot.bar(ax=ax, color=["#2874a6", "#cb4335"], rot=0, logy=True)
    ax.set(title="Label distribution (log count)", ylabel="BlockIds")
    for i, value in enumerate(distribution):
        ax.text(i, value * 1.1, f"{value:,}", ha="center")
    ax.set_ylim(1, distribution.max() * 3)
    save(fig, "label_distribution")

    for column, xlabel, name in [("length", "Events per BlockId", "trace_length"),
                                 ("latency", "Latency (unit unverified; symlog axis)", "latency")]:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for label, color in [("Normal", "#2874a6"), ("Anomaly", "#cb4335")]:
            values = frame.loc[frame.Label == label, column].dropna()
            # ECDF theo giá trị phân biệt, không vẽ nửa triệu điểm lặp.
            freq = values.value_counts().sort_index()
            ax.step(freq.index, freq.cumsum() / len(values), where="post", label=label, color=color)
        ax.set(xlabel=xlabel, ylabel="Fraction of BlockIds ≤ x", title=f"{column}: ECDF by class")
        if column == "latency":
            ax.set_xscale("symlog", linthresh=1)
        ax.legend()
        ax.grid(alpha=.2)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label in ("Normal", "Anomaly"):
        rates = frame.loc[frame.Label == label, "unique_events"].value_counts(normalize=True).sort_index()
        ax.plot(rates.index, rates, marker="o", label=label)
    ax.set(xlabel="Unique EventIds per BlockId", ylabel="Fraction within class", title="Unique events by class")
    ax.legend()
    save(fig, "unique_events")

    top = events.loc[events.prevalence_difference.abs().nlargest(12).index].sort_values("prevalence_difference")
    fig, ax = plt.subplots(figsize=(7, 4))
    top.prevalence_difference.plot.barh(ax=ax, color=["#cb4335" if x > 0 else "#2874a6" for x in top.prevalence_difference])
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(title="Largest event prevalence differences", xlabel="P(event | Anomaly) − P(event | Normal)", ylabel="EventId")
    save(fig, "event_prevalence_difference")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    maximum = int(patterns.blocks.max())
    ax.hist(patterns.blocks, bins=np.geomspace(1, maximum + 1, 30), color="#2874a6")
    ax.set(xscale="log", yscale="log", xlabel="BlockIds per exact-sequence group", ylabel="Number of groups", title="Duplicate group-size distribution")
    save(fig, "duplicate_group_sizes")
    return paths


def make_split_plot(composition: pd.DataFrame, output: Path) -> Path:
    """Nhận số liệu recompute từ verifier, không đọc producer report."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, protocol in zip(axes, ("random_block", "group_trace")):
        part = composition.loc[composition.protocol == protocol].set_index("split").reindex(["train", "validation", "test"])
        part[["Normal", "Anomaly"]].plot.bar(stacked=True, ax=ax, color=["#2874a6", "#cb4335"], rot=0)
        for i, row in enumerate(part.itertuples()):
            ax.text(i, row.blocks * 1.02, f"{row.block_fraction:.2%}", ha="center", fontsize=9)
        ax.set(title=protocol, ylabel="BlockIds", xlabel="", ylim=(0, part.blocks.max() * 1.2))
    fig.tight_layout()
    path = output / "figures" / "split_composition.png"
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
