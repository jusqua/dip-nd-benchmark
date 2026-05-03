import csv
import os
from math import floor, log

import matplotlib.pyplot as plot
import numpy as np

OUTPUT_DIR = "results"
TXT_FILENAME = "benchmark.txt"
CSV_FILENAME = "benchmark.csv"


def main():
    color_list = [
        "#00BFA0",
        "#B3D4FF",
        "#DC0AB4",
        "#FFA300",
        "#9B19F5",
        "#E6D800",
        "#50E991",
        "#0BB4FF",
        "#E60049",
    ]

    tech_results_dimensions_map: dict[str, dict[str, list[int]]] = {}
    tech_name_map: dict[str, str] = {}
    tech_color_map: dict[str, str] = {}
    group_results_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    single_results_map: dict[str, dict[str, list[float]]] = {}
    for tech in sorted(os.listdir(OUTPUT_DIR)):
        tech_path = os.path.join(OUTPUT_DIR, tech)
        if not os.path.isdir(tech_path):
            continue

        with open(os.path.join(tech_path, TXT_FILENAME)) as name:
            tech_name_map[tech] = name.read().strip()

        tech_results_dimensions_map[tech] = {}
        for dimension in sorted(os.listdir(tech_path)):
            results_path = os.path.join(tech_path, dimension)
            if not os.path.isdir(results_path):
                continue

            current_dimension = int("".join(filter(str.isdigit, dimension)))
            with open(os.path.join(results_path, CSV_FILENAME)) as results:
                reader = csv.reader(results)
                next(reader)
                for row in reader:
                    if tech_results_dimensions_map[tech].get(row[0]) is None:
                        tech_results_dimensions_map[tech][row[0]] = []
                    tech_results_dimensions_map[tech][row[0]].append(current_dimension)

                    if row[1] == "group":
                        if group_results_map.get(row[2]) is None:
                            group_results_map[row[2]] = {}
                        if group_results_map[row[2]].get(row[0]) is None:
                            group_results_map[row[2]][row[0]] = {}
                        if group_results_map[row[2]][row[0]].get(tech) is None:
                            group_results_map[row[2]][row[0]][tech] = []
                        group_results_map[row[2]][row[0]][tech].append(
                            float(row[-1]) * 1000000
                        )
                    if row[1] == "single":
                        if single_results_map.get(row[0]) is None:
                            single_results_map[row[0]] = {}
                        if single_results_map[row[0]].get(tech) is None:
                            single_results_map[row[0]][tech] = []
                        single_results_map[row[0]][tech].append(
                            float(row[-1]) * 1000000
                        )

    for operator, tech_map in single_results_map.items():
        _, ax = plot.subplots(figsize=(8, 6))

        result_lines = []
        line_titles = []
        dimensions = (1, 2, 3, 4, 5)
        dimensions_label = ("1D", "2D", "3D", "4D", "5D")

        for tech, results in tech_map.items():
            color = tech_color_map.get(tech)
            if color is None:
                color = tech_color_map[tech] = color_list.pop()
            result_lines.append(
                plot.plot(
                    tech_results_dimensions_map[tech][operator],
                    results,
                    color=color,
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
            yul = yll + 1
        ax.set_ylim(10**yll, 10**yul)

        plot.savefig(os.path.join(OUTPUT_DIR, f"{operator}.png"), bbox_inches="tight")
        plot.close()

    for group, results_map in group_results_map.items():
        _, ax = plot.subplots(figsize=(14, 7))

        operators = list(results_map.keys())
        technologies = set()
        for op_data in results_map.values():
            technologies.update(op_data.keys())
        technologies = sorted(list(technologies))

        x = np.arange(len(operators))
        width = 0.8 / len(technologies)

        bar_containers = []
        lowest_bar_height = float("inf")

        for idx, tech in enumerate(technologies):
            values = []
            for operator in operators:
                if tech in results_map[operator]:
                    avg_value = np.mean(results_map[operator][tech])
                    values.append(np.round(avg_value, 0))
                else:
                    values.append(0)

            color = tech_color_map.get(tech)
            if color is None:
                color = tech_color_map[tech] = color_list.pop()

            bars = ax.bar(
                x + idx * width,
                values,
                width,
                label=tech_name_map.get(tech, tech),
                color=color,
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
                        fontsize=12,
                    )

        ax.set_ylabel("Time (μs)", fontsize=14, weight="bold")
        ax.set_xticks(
            x + width * (len(technologies) - 1) / 2,
            [operator.title() for operator in operators],
            fontsize=14,
            weight="bold",
        )
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        yll, yul = (log(lim, 10) for lim in ax.get_ylim())
        if floor(yll) == floor(yul):
            yul = yll + 1
        ax.set_ylim(10**yll, 10**yul)

        plot.tight_layout()
        plot.savefig(
            os.path.join(OUTPUT_DIR, f"{group}-group.png"), bbox_inches="tight"
        )
        plot.close()


if __name__ == "__main__":
    main()
