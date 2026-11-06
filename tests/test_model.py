import numpy as np
import pytest

from darkstore import evaluate_sites, kmeans_sites, make_city, solve_design


@pytest.fixture(scope="module")
def city():
    return make_city(size_km=8, n_sites=15, seed=2)


@pytest.fixture(scope="module")
def design(city):
    return solve_design(city, radius_km=2.5)


def test_assignment_respects_constraints(city, design):
    a = design.assign
    assert np.all(a.sum(1) <= 1 + 1e-6), "a cell is served more than 100%"
    assert np.all(a[city.road_km > 2.5] == 0), "assignment beyond promise radius"
    closed = [j for j in range(len(city.sites)) if j not in design.open_sites]
    assert np.all(a[:, closed] == 0), "assignment to a closed store"
    load = (city.demand[:, None] * a).sum(0)
    assert np.all(load <= city.capacity + 1e-3), "store over capacity"


def test_milp_beats_or_matches_kmeans(city, design):
    km = evaluate_sites(city, kmeans_sites(city, len(design.open_sites)), 2.5)
    assert design.total <= km.total + 1e-3


def test_cost_breakdown_adds_up(city, design):
    s = design.summary(city)
    assert s["monthly_cost_lakh"] == pytest.approx(design.total / 1e5, abs=0.1)
    assert 0 < s["coverage_pct"] <= 100


def test_tighter_radius_never_serves_more(city):
    wide = solve_design(city, 3.5)
    tight = solve_design(city, 1.5)
    assert tight.served_orders(city) <= wide.served_orders(city) + 1e-6
    assert tight.total >= wide.total - 1e-3


def test_no_open_store_serves_nothing(city):
    d = evaluate_sites(city, [], 2.5)
    assert d.served_orders(city) == 0
    assert d.rent == 0
