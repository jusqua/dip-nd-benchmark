import csv
import os
import random

import matplotlib.pyplot as plot
import numpy as np

OUTPUT_DIR = "results"
TXT_FILENAME = "benchmark.txt"
CSV_FILENAME = "benchmark.csv"


def random_high_contrast_color():
    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        if luminance < 120:
            return (r, g, b)


def main():
    color_list = [
        "#FF1F3F",
        "#00B2A9",
        "#FFD700",
        "#4B0082",
        "#FF6F61",
        "#00FF7F",
        "#FF4500",
        "#1E90FF",
    ]

    tech_name_map: dict[str, str] = {}
    tech_color_map: dict[str, str] = {}
    group_results_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    single_results_map: dict[str, dict[str, list[float]]] = {}
    for tech in os.listdir(OUTPUT_DIR):
        tech_path = os.path.join(OUTPUT_DIR, tech)
        if not os.path.isdir(tech_path):
            continue

        with open(os.path.join(tech_path, TXT_FILENAME)) as name:
            tech_name_map[tech] = name.read().strip()

        for dimension in os.listdir(tech_path):
            results_path = os.path.join(tech_path, dimension)
            if not os.path.isdir(results_path):
                continue

            with open(os.path.join(results_path, CSV_FILENAME)) as results:
                reader = csv.reader(results)
                next(reader)
                for row in reader:
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
        fig, _ = plot.subplots(figsize=(8, 6))
        fig.add_subplot(111)

        result_lines = []
        line_titles = []
        dimensions = []
        dimensions_label = []

        for tech, results in tech_map.items():
            color = tech_color_map.get(tech)
            if not dimensions:
                dimensions = range(1, len(results) + 1)
                dimensions_label = [f"{i}D" for i in dimensions]
            if color is None:
                color = tech_color_map[tech] = (
                    color_list.pop(random.randint(0, len(color_list) - 1))
                    if color_list
                    else "#000000"
                )
            result_lines.append(plot.plot(dimensions, results, color=color, marker="o"))
            line_titles.append(tech_name_map[tech])

        plot.xlabel("Image Dimension", fontsize="large")
        plot.ylabel("Time (μs)", fontsize="large")
        plot.yscale("log")
        plot.xticks(dimensions, dimensions_label)
        plot.grid(True, axis="both")

        plot.legend(
            [x[0] for x in result_lines],
            line_titles,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=3,
        )

        plot.tight_layout()
        plot.savefig(os.path.join(OUTPUT_DIR, f"{operator}.png"))
        plot.close()

    for group, results_map in group_results_map.items():
        fig, ax = plot.subplots(figsize=(14, 7))

        operators = list(results_map.keys())
        technologies = set()
        for op_data in results_map.values():
            technologies.update(op_data.keys())
        technologies = sorted(list(technologies))

        x = np.arange(len(operators))
        width = 0.8 / len(technologies)

        bar_containers = []

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
                color = tech_color_map[tech] = (
                    color_list.pop(random.randint(0, len(color_list) - 1))
                    if color_list
                    else "#000000"
                )

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
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.0f}",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=9,
                    )

        ax.set_xlabel("Operations", fontsize="large", weight="bold")
        ax.set_ylabel("Time (μs)", fontsize="large", weight="bold")
        ax.set_xticks(x + width * (len(technologies) - 1) / 2)
        ax.set_xticklabels(operators, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=3,
        )

        plot.tight_layout()
        plot.savefig(
            os.path.join(OUTPUT_DIR, f"{group}-group.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plot.close()


if __name__ == "__main__":
    main()
