import argparse
import re
from typing import Literal

from colorama import Fore, Style, init
from scipy import stats


def parse_benchmark_file(
    file_path: str,
) -> dict[tuple[int, int], tuple[float, float, float]]:
    results: dict[tuple[int, int], tuple[float, float, float]] = {}
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(
                r"Size: (\d+), K: (\d+), Mean Time: ([\d.]+)s, CI: ([\d.]+)s - ([\d.]+)s",
                line,
            )
            if match:
                size, k = int(match.group(1)), int(match.group(2))
                mean_time = float(match.group(3))
                ci_lower, ci_upper = float(match.group(4)), float(match.group(5))
                results[(size, k)] = (mean_time, ci_lower, ci_upper)
    return results


def compare_benchmarks(old_file: str, new_file: str) -> None:
    old_results = parse_benchmark_file(old_file)
    new_results = parse_benchmark_file(new_file)

    init()  # Initialize colorama
    print(
        f"{'Size':>5} {'K':>3} {'Old (ms)':>10} {'New (ms)':>10} {'Change':>10} {'p-value':>10}"
    )
    print("-" * 55)

    for (size, k), (old_mean, old_lower, old_upper) in old_results.items():
        if (size, k) in new_results:
            new_mean, new_lower, new_upper = new_results[(size, k)]

            # Convert to milliseconds
            old_mean_ms = old_mean * 1000
            new_mean_ms = new_mean * 1000

            percent_change = (new_mean - old_mean) / old_mean * 100
            old_std = (old_upper - old_lower) / (2 * 1.96)
            new_std = (new_upper - new_lower) / (2 * 1.96)

            # Calculate degrees of freedom using Welch–Satterthwaite equation
            df = ((old_std**2 + new_std**2) ** 2) / (
                (old_std**4 / (30 - 1)) + (new_std**4 / (30 - 1))
            )

            t_statistic = (new_mean - old_mean) / ((old_std**2 + new_std**2) ** 0.5)
            p_value = stats.t.sf(abs(t_statistic), df) * 2

            color: Literal[Fore.GREEN, Fore.RED, ""] = ""
            if p_value < 0.05:
                color = Fore.GREEN if new_mean < old_mean else Fore.RED

            print(
                f"{size:5d} {k:3d} {old_mean_ms:10.2f} {new_mean_ms:10.2f} "
                f"{color}{percent_change:+10.2f}%{Style.RESET_ALL} {p_value:10.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark result files.")
    parser.add_argument("old_file", help="Path to the old benchmark results file")
    parser.add_argument("new_file", help="Path to the new benchmark results file")
    args = parser.parse_args()

    compare_benchmarks(args.old_file, args.new_file)


if __name__ == "__main__":
    main()
