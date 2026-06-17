import csv
import os
from math import floor, log

import matplotlib.pyplot as plot
import numpy as np

OUTPUT_DIR = "results"
TXT_FILENAME = "benchmark.txt"
CSV_FILENAME = "benchmark.csv"


def main():
    # Dutch field from https://www.geeksforgeeks.org/data-visualization/color-palettes-for-data-visualization/
    color_list = [
        "#E6D800",
        "#B3D4FF",
        "#9B19F5",
        "#DC0AB4",
        "#0BB4FF",
        "#E60049",
        "#FFA300",
        "#50E991",
        "#00BFA0",
    ]

    tech_name_map: dict[str, str] = {}
    tech_color_map: dict[str, str] = {}
    rgroup_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    rsingle_map: dict[str, dict[str, dict[int, list[float]]]] = {}
    for tech in sorted(os.listdir(OUTPUT_DIR)):
        tech_path = os.path.join(OUTPUT_DIR, tech)
        if not os.path.isdir(tech_path):
            continue

        with open(os.path.join(tech_path, TXT_FILENAME)) as name:
            tech_name_map[tech] = name.read().strip()

        for dimension in sorted(os.listdir(tech_path)):
            results_path = os.path.join(tech_path, dimension)
            if not os.path.isdir(results_path):
                continue

            dimension = int("".join(filter(str.isdigit, dimension)))
            with open(os.path.join(results_path, CSV_FILENAME)) as results:
                reader = csv.reader(results)
                next(reader)
                for row in reader:
                    operator, rtype, group, duration = row

                    if rtype == "group":
                        if rgroup_map.get(group) is None:
                            rgroup_map[group] = {}
                        if rgroup_map[group].get(operator) is None:
                            rgroup_map[group][operator] = {}
                        if rgroup_map[group][operator].get(tech) is None:
                            rgroup_map[group][operator][tech] = []
                        rgroup_map[group][operator][tech].append(np.float64(duration))
                    if rtype == "single":
                        if rsingle_map.get(operator) is None:
                            rsingle_map[operator] = {}
                        if rsingle_map[operator].get(tech) is None:
                            rsingle_map[operator][tech] = {}
                        if rsingle_map[operator][tech].get(dimension) is None:
                            rsingle_map[operator][tech][dimension] = []
                        rsingle_map[operator][tech][dimension].append(
                            np.float64(duration)
                        )

    for tech in tech_name_map.keys():
        tech_color_map[tech] = color_list.pop()

    for operator, tech_map in rsingle_map.items():
        _, ax = plot.subplots(figsize=(8, 6))

        result_lines = []
        line_titles = []
        dimensions = (1, 2, 3, 4, 5)
        dimensions_label = ("1D", "2D", "3D", "4D", "5D")

        for tech, results in tech_map.items():
            result_lines.append(
                plot.plot(
                    list(results.keys()),
                    [np.mean(v) * 1_000_000 for v in results.values()],
                    color=tech_color_map.get(tech),
                    marker="o",
                )
            )
            line_titles.append(tech_name_map[tech])

        ax.set_ylabel("Time (μs)", fontsize=14, weight="bold")
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xticks(dimensions, dimensions_label, fontsize=12)
        ax.grid(True, axis="both", alpha=0.3)
        ax.legend([x[0] for x in result_lines], line_titles)
        yll, yul = (log(lim, 10) for lim in ax.get_ylim())
        if floor(yll) == floor(yul):
            yll = 0 if yll - 1 < 0 else yll - 1
        ax.set_ylim(bottom=10**yll)

        plot.savefig(os.path.join(OUTPUT_DIR, f"{operator}.png"), bbox_inches="tight")
        plot.close()

    for group, op_map in rgroup_map.items():
        _, ax = plot.subplots(figsize=(8, 6))

        ops = list(op_map.keys())
        techs = set()
        for tech_map in op_map.values():
            techs.update(tech_map.keys())
        techs = sorted(list(techs))

        x = np.arange(len(ops))
        width = 0.8 / len(techs)
        bar_containers = []
        lowest_bar_height = float("inf")

        for idx, tech in enumerate(techs):
            bars = ax.bar(
                x + idx * width,
                [np.round(np.mean(op_map[op][tech]) * 1_000_000, 0) for op in ops],
                width,
                label=tech_name_map.get(tech, tech),
                color=tech_color_map.get(tech),
                alpha=0.8,
            )
            bar_containers.append(bars)

            for bar in bars:
                height = bar.get_height()
                if height < lowest_bar_height:
                    lowest_bar_height = height

        for bars in bar_containers:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        lowest_bar_height + (lowest_bar_height * 0.1),
                        f"{height:.0f}",
                        ha="center",
                        va="bottom",
                        weight="bold",
                        rotation=90,
                        fontsize=10,
                    )

        ax.set_ylabel("Time (μs)", fontsize=14, weight="bold")
        ax.set_xticks(
            x + width * (len(techs) - 1) / 2,
            [operator.title() for operator in ops],
            fontsize=14,
            weight="bold",
        )
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        yll, yul = (log(lim, 10) for lim in ax.get_ylim())
        if floor(yll) == floor(yul):
            yll = 0 if yll - 1 < 0 else yll - 1
        ax.set_ylim(bottom=10**yll)

        plot.tight_layout()
        plot.savefig(
            os.path.join(OUTPUT_DIR, f"{group}-group.png"), bbox_inches="tight"
        )
        plot.close()


if __name__ == "__main__":
    main()
