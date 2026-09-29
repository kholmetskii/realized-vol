"""Plot real-data volatility signatures and microstructure-noise diagnostics.

Usage:
    python scripts/diagnostics_plots.py \
        data/EURUSD_2024-01-01_2024-03-31.parquet \
        --out figures/diagnostics_overview.png
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rvol.diagnostics.microstructure import noise_test  # noqa: E402
from rvol.diagnostics.signature import signature  # noqa: E402
from rvol.plotting.diagnostics import plot_diagnostics_overview  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", help="tick file produced by rvol.data.dukascopy")
    parser.add_argument("--out", default="figures/diagnostics_overview.png")
    parser.add_argument("--fine", default="1s")
    parser.add_argument("--coarse", default="5min")
    args = parser.parse_args()

    ticks = pd.read_parquet(args.parquet)
    columns = ("mid", "bid", "ask")
    signatures = {column: signature(ticks, column) for column in columns}
    noise_tests = {
        column: noise_test(
            ticks,
            column,
            fine=args.fine,
            coarse=args.coarse,
        )
        for column in columns
    }

    name = pathlib.Path(args.parquet).stem
    figure = plot_diagnostics_overview(
        signatures,
        noise_tests,
        title=f"Realized-volatility diagnostics — {name}",
    )
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    for column, result in noise_tests.items():
        print(f"--- {column}\n{result}\n")
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
