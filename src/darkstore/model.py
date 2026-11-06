"""Capacitated facility location with a delivery-promise radius.

    min   sum_j rent_j y_j
        + 30 * sum_ij demand_i * c_ij * x_ij          (rider pay per order)
        + 30 * sum_i  demand_i * m * (1 - sum_j x_ij)   (lost margin)
    s.t.  sum_j x_ij <= 1                              each cell served at most once
          x_ij <= y_j                                  only from open stores
          sum_i demand_i x_ij <= cap_j y_j             store throughput
          x_ij = 0  if road_km_ij > R                  10-minute promise
          y_j binary, 0 <= x_ij <= 1

x_ij is the share of cell i's orders fulfilled by store j. Everything is
monthly INR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from .data import City

DAYS = 30


@dataclass(frozen=True)
class Economics:
    base_cost: float = 18.0        # INR per order: pick/pack + rider base fare
    per_km: float = 9.0            # INR per road-km (there and back folded in)
    lost_margin: float = 90.0      # INR lost per unserved order: margin + churn risk


@dataclass
class Design:
    name: str
    radius_km: float
    open_sites: list[int]
    assign: np.ndarray             # (I, J) fractions
    rent: float
    delivery: float
    lost: float

    @property
    def total(self) -> float:
        return self.rent + self.delivery + self.lost

    def served_orders(self, city: City) -> float:
        return float((city.demand[:, None] * self.assign).sum())

    def summary(self, city: City) -> dict:
        served = self.served_orders(city)
        dist = (city.demand[:, None] * self.assign * city.road_km).sum() / max(served, 1)
        util = [float((city.demand * self.assign[:, j]).sum() / city.capacity[j])
                for j in self.open_sites]
        return {
            "design": self.name,
            "radius_km": self.radius_km,
            "stores": len(self.open_sites),
            "coverage_pct": round(100 * served / city.total_demand, 1),
            "avg_delivery_km": round(float(dist), 2),
            "avg_utilisation_pct": round(100 * float(np.mean(util)), 1) if util else 0.0,
            "monthly_cost_lakh": round(self.total / 1e5, 1),
            "cost_per_served_order": round((self.rent + self.delivery) / (DAYS * max(served, 1)), 1),
        }


def _build(city: City, radius_km: float, econ: Economics, fixed_open=None):
    I, J = city.road_km.shape
    unit = econ.base_cost + econ.per_km * city.road_km
    allowed = city.road_km <= radius_km
    pairs = [(i, j) for i in range(I) for j in range(J) if allowed[i, j] and city.demand[i] > 0]

    m = pulp.LpProblem("dark_store_design", pulp.LpMinimize)
    if fixed_open is None:
        y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(J)}
    else:
        y = {j: (1 if j in fixed_open else 0) for j in range(J)}
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0, 1) for i, j in pairs}

    by_cell: dict[int, list] = {}
    by_site: dict[int, list] = {}
    for i, j in pairs:
        by_cell.setdefault(i, []).append(x[i, j])
        by_site.setdefault(j, []).append((i, x[i, j]))

    served = {i: pulp.lpSum(v) for i, v in by_cell.items()}
    m += (pulp.lpSum(city.rent[j] * y[j] for j in range(J))
          + DAYS * pulp.lpSum(city.demand[i] * unit[i, j] * x[i, j] for i, j in pairs)
          + DAYS * econ.lost_margin * pulp.lpSum(city.demand[i] * (1 - served.get(i, 0))
                                                 for i in range(I) if city.demand[i] > 0))
    for i, v in by_cell.items():
        m += pulp.lpSum(v) <= 1
    for (i, j), var in x.items():
        if fixed_open is None:
            m += var <= y[j]
        elif j not in fixed_open:
            var.upBound = 0
    for j, lst in by_site.items():
        m += pulp.lpSum(city.demand[i] * v for i, v in lst) <= city.capacity[j] * y[j]
    return m, x, y, unit


def _extract(city, name, radius_km, econ, x, open_sites, unit) -> Design:
    I, J = city.road_km.shape
    assign = np.zeros((I, J))
    for (i, j), var in x.items():
        assign[i, j] = var.value() or 0.0
    assign[assign < 1e-6] = 0.0
    rent = float(sum(city.rent[j] for j in open_sites))
    delivery = float(DAYS * (city.demand[:, None] * unit * assign).sum())
    lost = float(DAYS * econ.lost_margin * (city.demand * (1 - assign.sum(1))).sum())
    return Design(name, radius_km, sorted(open_sites), assign, rent, delivery, lost)


def solve_design(city: City, radius_km: float = 2.5, econ: Economics = Economics(),
                 time_limit_s: int = 120, gap: float = 0.005) -> Design:
    m, x, y, unit = _build(city, radius_km, econ)
    status = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, gapRel=gap))
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved") and m.status != 1:
        raise RuntimeError(f"solver status {pulp.LpStatus[status]}")
    open_sites = [j for j, v in y.items() if (v.value() or 0) > 0.5]
    return _extract(city, "MILP optimum", radius_km, econ, x, open_sites, unit)


def evaluate_sites(city: City, open_sites, radius_km: float, econ: Economics = Economics(),
                   name: str = "fixed sites") -> Design:
    """Best possible assignment (LP) for a *given* set of stores."""
    open_sites = set(open_sites)
    m, x, _, unit = _build(city, radius_km, econ, fixed_open=open_sites)
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return _extract(city, name, radius_km, econ, x, open_sites, unit)


def kmeans_sites(city: City, k: int, seed: int = 0) -> list[int]:
    """Planner's heuristic: demand-weighted k-means, snapped to the nearest
    free candidate site."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(city.cells, sample_weight=city.demand)
    chosen: list[int] = []
    for c in km.cluster_centers_:
        order = np.argsort(np.hypot(*(city.sites - c).T))
        chosen.append(next(int(j) for j in order if j not in chosen))
    return chosen
