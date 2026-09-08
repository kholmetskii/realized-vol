"""Plot the volatility signature for a tick file.

Usage:
    python scripts/signature_plot.py data/EURUSD_2024-01-15_2024-03-15.parquet
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rvol.estimators.signature import signature  # noqa: E402


def plot(sig: pd.DataFrame, out: pathlib.Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sig["seconds"], sig["ann_vol_pct"], "o-", lw=1.5, ms=5)
    ax.axvline(300, ls="--", lw=1, color="grey")
    ax.annotate("5 min", xy=(300, ax.get_ylim()[0]), xytext=(320, ax.get_ylim()[0]),
                fontsize=9, color="grey", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("sampling interval, s (log scale)")
    ax.set_ylabel("annualised volatility from RV, %")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"figure: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", help="tick file produced by rvol.data.dukascopy")
    p.add_argument("--price", default="mid", choices=["mid", "bid", "ask"])
    p.add_argument("--out", default="figures/signature_plot.png")
    args = p.parse_args()

    ticks = pd.read_parquet(args.parquet)
    sig = signature(ticks, args.price)
    print(sig.to_string(index=False))

    ratio = sig["mean_RV"].iloc[0] / sig.loc[sig["freq"] == "5min", "mean_RV"].iloc[0]
    print(f"\nRV(5s) / RV(5min) = {ratio:.2f}"
          "  — how far noise inflates the estimate at tick scale")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(sig, out, f"Volatility signature plot — {pathlib.Path(args.parquet).stem}")


if __name__ == "__main__":
    main()
