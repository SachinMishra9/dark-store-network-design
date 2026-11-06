"""Synthetic city: a demand grid plus candidate dark-store sites.

* Demand: orders/day per 1 km x 1 km cell, from a mixture of population
  hot-spots (dense residential pockets) on top of a thin background.
* Candidates: leasable sites with monthly rent that rises toward the
  city centre and a daily order-handling capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CIRCUITY = 1.3


@dataclass
class City:
    cells: np.ndarray        # (I, 2) km, centre of each demand cell
    demand: np.ndarray       # (I,) orders/day
    sites: np.ndarray        # (J, 2) km, candidate locations
    rent: np.ndarray         # (J,) INR / month (rent + fixed staffing)
    capacity: np.ndarray     # (J,) orders/day
    road_km: np.ndarray      # (I, J) road distance cell -> site

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())


def make_city(size_km: int = 16, n_hotspots: int = 7, n_sites: int = 45,
              seed: int = 0) -> City:
    rng = np.random.default_rng(seed)
    half = size_km / 2
    g = np.arange(-half + 0.5, half, 1.0)
    cells = np.array([(x, y) for x in g for y in g])

    hot = rng.uniform(-0.65 * half, 0.65 * half, size=(n_hotspots, 2))
    weight = rng.uniform(0.5, 1.5, size=n_hotspots)
    spread = rng.uniform(1.2, 2.5, size=n_hotspots)
    d2 = ((cells[:, None, :] - hot[None, :, :]) ** 2).sum(-1)
    density = (weight * np.exp(-d2 / (2 * spread ** 2))).sum(1) + 0.04
    demand = np.round(density / density.sum() * 18000 * rng.lognormal(0, 0.15, len(cells)))

    sites = rng.uniform(-0.9 * half, 0.9 * half, size=(n_sites, 2))
    r = np.hypot(sites[:, 0], sites[:, 1])
    rent = np.round(250_000 + 400_000 * np.exp(-(r ** 2) / (2 * 4.0 ** 2)), -3)
    capacity = rng.choice([1500, 2500, 3500], size=n_sites, p=[0.4, 0.4, 0.2]).astype(float)
    rent = rent * (0.75 + capacity / 10_000)   # bigger box, bigger lease

    road_km = np.hypot(*(cells[:, None, :] - sites[None, :, :]).transpose(2, 0, 1)) * CIRCUITY
    return City(cells, demand, sites, rent, capacity, road_km)
