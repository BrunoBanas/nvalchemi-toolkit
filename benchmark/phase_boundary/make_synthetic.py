# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic SGC isotherms with a KNOWN miscibility gap, for validating sgc_phase_boundary.py.

Rigid-lattice mean-field sub-regular solution (x = x_B, per atom, eV):
    E(x) = x(1-x)[L0 + L1(1-2x)] + eB x,  S = ideal,  F_A = 0,  F_B = eB
    dmu(x) = dG/dx.
Each branch follows its own (metastable) root and jumps only at its spinodal, as a real
SGC walker would with slow nucleation. Writes a tidy CSV plus truth.json (common-tangent
binodal on a dense grid) and pure.json (F_A, F_B for --pure-free-energies).

    python make_synthetic.py out_dir [--design overlap|meet|single]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

KB = 8.617333262e-5


def model(L0, L1, eB, T, x):
    """Return excess energy, free energy and exchange potential of the sub-regular model at *x*."""
    Ex = x * (1 - x) * (L0 + L1 * (1 - 2 * x)) + eB * x
    G = Ex + KB * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
    dEx = (1 - 2 * x) * (L0 + L1 * (1 - 2 * x)) - 2 * L1 * x * (1 - x) + eB
    mu = dEx + KB * T * np.log(x / (1 - x))
    return Ex, G, mu


def truth(L0, L1, eB, T):
    """Return the exact binodal at *T* from the convex hull of G(x), or ``None`` above T_c."""
    x = np.linspace(1e-6, 1 - 1e-6, 200001)
    _, G, mu = model(L0, L1, eB, T, x)
    hull = [0]
    for i in range(1, len(x)):
        while (
            len(hull) >= 2
            and (x[hull[-1]] - x[hull[-2]]) * (G[i] - G[hull[-2]])
            - (G[hull[-1]] - G[hull[-2]]) * (x[i] - x[hull[-2]])
            <= 0
        ):
            hull.pop()
        hull.append(i)
    gaps = np.diff(x[hull])
    k = int(np.argmax(gaps))
    if gaps[k] < 1e-3:
        return None
    i, j = hull[k], hull[k + 1]
    return dict(
        x_alpha=float(x[i]),
        x_gamma=float(x[j]),
        mu_coex=float((G[j] - G[i]) / (x[j] - x[i])),
    )


def branch_x(L0, L1, eB, T, mus, start_low):
    """Follow one metastable branch across the *mus* ladder, jumping only past its spinodal."""
    x = np.linspace(1e-7, 1 - 1e-7, 400001)
    _, _, mu = model(L0, L1, eB, T, x)
    dmu = np.diff(mu)
    stable = np.concatenate([[True], dmu > 0])
    # contiguous increasing pieces of mu(x): low piece starts at x=0, high piece ends at x=1
    breaks = np.where(~stable)[0]
    lo_end = breaks[0] if len(breaks) else len(x)
    hi_start = breaks[-1] + 1 if len(breaks) else 0
    lo_x, lo_mu, hi_x, hi_mu = x[:lo_end], mu[:lo_end], x[hi_start:], mu[hi_start:]
    out, on_low = [], start_low
    for m in mus:
        if on_low and m > lo_mu[-1]:
            on_low = False
        if not on_low and m < hi_mu[0]:
            on_low = True
        out.append(
            float(np.interp(m, lo_mu, lo_x))
            if on_low
            else float(np.interp(m, hi_mu, hi_x))
        )
    return np.array(out)


def main():
    """Write synthetic SGC isotherms with known boundaries to the output directory."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument(
        "--design",
        choices=["overlap", "meet"],
        default="overlap",
        help="overlap: both branches sweep the full window; meet: ladders stop at the centre",
    )
    ap.add_argument("--L0", type=float, default=0.20)
    ap.add_argument("--L1", type=float, default=0.03)
    ap.add_argument("--eB", type=float, default=-0.5)
    ap.add_argument(
        "--temperatures", type=float, nargs="+", default=[800, 950, 1050, 1300]
    )
    ap.add_argument("--noise", type=float, default=0.004)
    ap.add_argument("--half-width", type=float, default=0.30)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows, tr, pure = [], {}, {}
    for T in a.temperatures:
        t = truth(a.L0, a.L1, a.eB, T)
        tr[f"{T:g}"] = t
        pure[f"{T:g}"] = {"A": 0.0, "B": a.eB}
        centre = t["mu_coex"] if t else a.eB
        grid = np.round(
            np.arange(centre - a.half_width, centre + a.half_width + 1e-9, a.step), 6
        )
        for name, mus, start_low in (("low", grid, True), ("high", grid[::-1], False)):
            if a.design == "meet":
                mus = (
                    mus[mus <= centre + 1e-9]
                    if start_low
                    else mus[mus >= centre - 1e-9]
                )
            xs = branch_x(a.L0, a.L1, a.eB, T, mus, start_low)
            for k, (m, x) in enumerate(zip(mus, xs)):
                E, _, _ = model(a.L0, a.L1, a.eB, T, np.array([x]))
                xn = float(np.clip(x + rng.normal(0, a.noise), 1e-5, 1 - 1e-5))
                rows.append(
                    dict(
                        T=T,
                        branch=name,
                        order=k,
                        mu=float(m),
                        x=xn,
                        x_se=a.noise,
                        x_drift=0.0,
                        E=float(E[0]) + rng.normal(0, 0.002),
                        E_se=0.002,
                        resolved=True,
                    )
                )
    with open(os.path.join(a.out, "synthetic.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(tr, open(os.path.join(a.out, "truth.json"), "w"), indent=2)
    json.dump(pure, open(os.path.join(a.out, "pure.json"), "w"), indent=2)
    print(json.dumps(tr, indent=2))


if __name__ == "__main__":
    main()
