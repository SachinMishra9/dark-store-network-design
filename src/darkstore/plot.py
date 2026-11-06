from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import City  # noqa: E402
from .model import Design  # noqa: E402


def plot_designs(city: City, designs: list[Design], path: str) -> None:
    fig, axes = plt.subplots(1, len(designs), figsize=(6.2 * len(designs), 6), squeeze=False)
    half = np.abs(city.cells).max() + 0.5
    for ax, d in zip(axes[0], designs):
        served = d.assign.sum(1)
        sc = ax.scatter(city.cells[:, 0], city.cells[:, 1], c=city.demand, cmap="YlOrRd",
                        s=95, marker="s", edgecolors="none")
        lost = served < 0.99
        ax.scatter(city.cells[lost, 0], city.cells[lost, 1], marker="x", c="#333", s=14,
                   lw=0.8, label="not (fully) served")
        ax.scatter(city.sites[:, 0], city.sites[:, 1], marker="o", facecolors="none",
                   edgecolors="#777", s=30, label="candidate site")
        for j in d.open_sites:
            ax.add_patch(plt.Circle(city.sites[j], d.radius_km / 1.3, fill=False, ls="--",
                                    color="#1f4e9c", lw=0.8))
        o = city.sites[d.open_sites]
        ax.scatter(o[:, 0], o[:, 1], marker="*", c="#1f4e9c", s=city.capacity[d.open_sites] / 12,
                   edgecolors="white", zorder=3, label="open dark store")
        s = d.summary(city)
        ax.set_title(f"{d.name}: {s['stores']} stores, {s['coverage_pct']}% of orders served\n"
                     f"INR {s['cost_per_served_order']}/order, total INR {s['monthly_cost_lakh']} lakh/month",
                     fontsize=10)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xlabel("km")
        ax.legend(loc="lower left", fontsize=7)
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.7, pad=0.02, label="orders / day / km²")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(rows: list[dict], path: str) -> None:
    r = [x["radius_km"] for x in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(r, [x["stores"] for x in rows], "o-", color="#1f4e9c", label="stores opened")
    ax1.set_xlabel("delivery promise radius (road km from store)")
    ax1.set_ylabel("stores opened", color="#1f4e9c")
    ax2 = ax1.twinx()
    ax2.plot(r, [x["coverage_pct"] for x in rows], "s--", color="#2a9d55", label="orders served %")
    ax2.plot(r, [x["avg_utilisation_pct"] for x in rows], "^:", color="#d1495b", label="store utilisation %")
    ax2.set_ylabel("%")
    ax2.set_ylim(0, 105)
    for x in rows:
        ax1.annotate(f"INR {x['cost_per_served_order']:.0f}/order", (x["radius_km"], x["stores"]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    ax1.set_title("Cost of a faster delivery promise")
    ax1.grid(alpha=0.25)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
