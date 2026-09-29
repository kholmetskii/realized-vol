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

from rvol.diagnostics.signature import signature  # noqa: E402
from rvol.plotting.diagnostics import plot_volatility_signature  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquet", help="tick file produced by rvol.data.dukascopy")
    p.add_argument("--price", default="mid", choices=["mid", "bid", "ask"])
    p.add_argument("--out", default="figures/signature_plot.png")
    args = p.parse_args()

    ticks = pd.read_parquet(args.parquet)
    sig = signature(ticks, args.price)
    print(sig.to_string(index=False))

    finest = sig.iloc[0]
    five_min = sig.loc[sig["freq"] == "5min"].iloc[0]
    ratio = finest["mean_RV"] / five_min["mean_RV"]
    print(f"\nsessions used: {int(five_min['n_days'])}")
    print(f"RV({finest['freq']}) / RV(5min) = {ratio:.2f}"
          "  — how far noise inflates the estimate at tick scale")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_volatility_signature(
        {args.price: sig},
        title=f"Volatility signature — {pathlib.Path(args.parquet).stem}",
    )
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
