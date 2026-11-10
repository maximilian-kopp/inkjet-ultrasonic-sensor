#!/usr/bin/env python3
"""Checks on numbers stated in the inkjet SHM manuscript CSVs.

This script does not digitize plots. It verifies arithmetic that can be
formed from tabulated or in-text values, and it attempts to recover the
stated defect coordinates from the stated time delays using a standard
time-of-flight path-length model (direct + reflected minus the direct
source-to-sensor path; source echo treated as a two-way path). That
model is the natural reading of the prose in §VI; the explicit
equations are printed as images in the manuscript and were not retyped.

Usage, from the repo or package root:

    python src/reproduce_inkjet_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IN_PER_M = 39.37007874015748
U_MPS = 5000.0  # speed of sound in aluminum, as stated in §VI
U_IPS = U_MPS * IN_PER_M


def load_table(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader((ln for ln in f if not ln.startswith("#"))))


def residuals(D, S, sensors, dts_s, dtS_s=None):
    D = np.asarray(D, float)
    S = np.asarray(S, float)
    res = []
    for M, dt in zip(sensors, dts_s):
        M = np.asarray(M, float)
        pred = np.linalg.norm(D - S) + np.linalg.norm(D - M) - np.linalg.norm(S - M)
        res.append(pred - dt * U_IPS)
    if dtS_s is not None:
        res.append(2 * np.linalg.norm(D - S) - dtS_s * U_IPS)
    return np.array(res)


def fit(src, sensors, dts_us, dtS_us, x0, niter=80):
    D = np.array(x0, float)
    S = np.array(src, float)
    dts = np.array(dts_us, float) * 1e-6
    dtS = None if dtS_us is None else dtS_us * 1e-6
    for _ in range(niter):
        r = residuals(D, S, sensors, dts, dtS)
        J = np.zeros((len(r), 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = 1e-7
            J[:, k] = (residuals(D + e, S, sensors, dts, dtS) - r) / 1e-7
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        D = D + step
        if np.linalg.norm(step) < 1e-12:
            break
    r = residuals(D, S, sensors, dts, dtS)
    return D, float(np.dot(r, r))


def parse_coord(s: str):
    s = s.strip().replace("−", "-")
    x, y = s.strip("() ").split(",")
    return (float(x), float(y))


def main() -> int:
    prices = load_table(DATA / "inkjet-table3-commercial-comparison.csv")
    loc = load_table(DATA / "inkjet-table4-localization-results.csv")
    first = load_table(DATA / "inkjet-first-attempt-localization.csv")[0]

    print("Inkjet SHM manuscript: stated-number checks")
    print("-------------------------------------------")

    this_work = next(r for r in prices if r["brand_model"] == "This work")
    commercials = [r for r in prices if r["brand_model"] != "This work"]
    stated_price = float(this_work["price_usd"])
    min_commercial = min(float(r["price_usd"]) for r in commercials)
    reduction = (1.0 - stated_price / min_commercial) * 100
    print(f"Table 3 price, this work:           ${stated_price}")
    print(f"Lowest commercial price in Table 3: ${min_commercial:.0f} (GE SEB4-EN)")
    print(f"1 - 0.1/900:                        {reduction:.3f}%")
    print("The manuscript text (conclusion) states a reduction of nearly 99%.")
    print(f"Table 3 supports a reduction of at least 99% vs the listed commercial prices: {reduction >= 99}")
    print()

    # First-attempt 0.3% : distance from stated (Xd,Yd) to actual (0,-4)
    xd = float(first["stated_Xd_in"])
    yd = float(first["stated_Yd_in"])
    actual = (0.0, -4.0)
    dist = float(np.hypot(xd - actual[0], yd - actual[1]))
    # The manuscript states "a measurement error of only 0.3%" without
    # writing the formula. Distance / 7.2 in (the stated S1=S2) is 0.311%.
    s12 = float(first["S1_S2_in"])
    pct_vs_s = dist / s12 * 100
    print("First-attempt stated result (§VI):")
    print(f"  Stated (Xd, Yd) = ({xd}, {yd}) in; actual (0, -4) in")
    print(f"  Distance from actual:             {dist:.5f} in")
    print(f"  Distance / 7.2 in × 100%:         {pct_vs_s:.3f}%")
    print(f"  Rounds to the stated 0.3%:        {round(pct_vs_s, 1) == 0.3}")
    print()

    print("Time-of-flight recovery of stated (Xd, Yd)")
    print("  Model: extra path = |D-S| + |D-M| - |S-M| = u Δt, u = 5000 m/s;")
    print("  source echo (when a ΔtS is given) = 2|D-S| / u.")
    print()

    # first attempt: two sensors, Δt2=Δt3=13.9 μs; Δt1=80.2 μs is also stated
    src = parse_coord(first["source_coord_in"])
    m1 = parse_coord(first["sensor1_coord_in"])
    m2 = parse_coord(first["sensor2_coord_in"])
    dt2 = float(first["delta_t2_us"])
    dt3 = float(first["delta_t3_us"])
    dt1 = float(first["delta_t1_us"])
    D_two, cost_two = fit(src, [m1, m2], [dt2, dt3], None, (0, -4))
    D_with, cost_with = fit(src, [m1, m2], [dt2, dt3], dt1, (0, -4))
    stated_xy = np.array([xd, yd])
    d_two = float(np.linalg.norm(D_two - stated_xy))
    d_with = float(np.linalg.norm(D_with - stated_xy))
    print("  First attempt, using only Δt2=Δt3=13.9 μs (no source echo):")
    print(f"    fit (Xd, Yd) = ({D_two[0]:.4f}, {D_two[1]:.4f}) in")
    print(f"    distance from stated (−0.02, −4.01): {d_two:.4f} in")
    print("  First attempt, also treating Δt1=80.2 μs as a two-way source echo:")
    print(f"    fit (Xd, Yd) = ({D_with[0]:.4f}, {D_with[1]:.4f}) in")
    print(f"    distance from stated (−0.02, −4.01): {d_with:.4f} in")
    first_ok = d_two < 0.05  # tight match not expected; record outcome
    print()

    print("  Table 4 rows:")
    row_matches = []
    for row in loc:
        src = parse_coord(row["source_in"])
        s1 = parse_coord(row["sensor1_in"])
        s2 = parse_coord(row["sensor2_in"])
        dts = [float(row["delta_t1_us"]), float(row["delta_t2_us"])]
        dtS = float(row["delta_tS_us"])
        stated = (float(row["Xd_in"]), float(row["Yd_in"]))
        # start near the plate center and also near the stated Y sign
        D, cost = fit(src, [s1, s2], dts, dtS, (0.0, stated[1]))
        err = float(np.linalg.norm(D - np.array(stated)))
        close = err < 0.05
        row_matches.append(close)
        print(
            f"    Row {row['trial']}: fit=({D[0]:.4f}, {D[1]:.4f}) "
            f"stated=({stated[0]}, {stated[1]})  |Δ|={err:.4f} in  close={close}"
        )
    print()
    print("  The standard ToF model does not recover every stated (Xd, Yd)")
    print("  within 0.05 in. Table 4 coordinates are therefore reported as")
    print("  written in the manuscript, not as values this script refits.")
    print()
    print("38% sensitivity")
    print("---------------")
    print("The abstract and conclusion state that the printed sensors")
    print("outperform commercial ultrasonic (or triboelectric) sensors by")
    print("approximately 38% in sensitivity. No table in the manuscript")
    print("lists the two sensitivity values used to form that percentage,")
    print("so it cannot be recomputed from the CSVs.")
    print()

    # Do not fail the script solely because ToF recovery is imperfect —
    # that is a reported finding, not a packaging error. Fail only if
    # the clearly defined arithmetic (cost reduction, 0.3% vs 7.2 in) breaks.
    pct_ok = round(pct_vs_s, 1) == 0.3
    cost_ok = reduction >= 99
    ok = pct_ok and cost_ok
    print(
        "RESULT:",
        "PASS (price reduction and 0.3% distance/7.2 in arithmetic match); "
        "ToF coordinate recovery did not fully match the stated (Xd, Yd)."
        if ok
        else "FAIL",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
