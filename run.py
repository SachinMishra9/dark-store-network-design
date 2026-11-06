"""Reproduce the README results.

    python run.py           # ~10 s
    python run.py --quick   # CI smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darkstore import evaluate_sites, kmeans_sites, make_city, solve_design
from darkstore.plot import plot_designs, plot_tradeoff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    city = make_city(size_km=10 if a.quick else 16, n_sites=20 if a.quick else 45, seed=a.seed)

    best = solve_design(city, a.radius)
    km = evaluate_sites(city, kmeans_sites(city, len(best.open_sites), a.seed), a.radius,
                        name="k-means placement")
    plot_designs(city, [km, best], str(out / "network_map.png"))

    radii = [2.0, 3.0] if a.quick else [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    sweep = [solve_design(city, r).summary(city) for r in radii]
    plot_tradeoff(sweep, str(out / "radius_tradeoff.png"))

    result = {"radius_km": a.radius, "comparison": [km.summary(city), best.summary(city)],
              "radius_sweep": sweep}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    keys = ["stores", "coverage_pct", "avg_delivery_km", "avg_utilisation_pct",
            "cost_per_served_order", "monthly_cost_lakh"]
    print(f"{'':22s}" + "".join(f"{k:>22s}" for k in keys))
    for s in result["comparison"]:
        print(f"{s['design']:22s}" + "".join(f"{s[k]:>22}" for k in keys))
    print("\nradius sweep (MILP)")
    for s in sweep:
        print(f"R={s['radius_km']:<4} " + "  ".join(f"{k}={s[k]}" for k in keys))


if __name__ == "__main__":
    main()
