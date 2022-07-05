import os
from typing import List
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import seaborn
import warnings

import plotly.express as px
import plotly.graph_objects as go


def plot_all_to_all(
    group: str,
    crop: str,
    organelle_name: str,
    metric_value: str,
    metric_name: str,
    segmentation_types: List[str],
    all_to_all: np.ndarray,
    score_range: dict,
    output_path: str,
) -> None:
    """Plot the output all-to-all matrix for a given set of data.

    Args:
        group (str): Group to plot.
        crop (str): Crop to plot.
        organelle_name (str): Organelle to plot.
        metric_value (str): Metric value (contains underscores).
        metric_name (str): Metric name.
        segmentation_types (List[str]): List of all segmentations (annotations, predictions, refinements, ariadne, ...)
        all_to_all (np.ndarray): All-to-all matrix containing scores.
        score_range (dict): Dict containing information relevant to score plotting.
        output_path (str): Path to output images.
    """
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"

    # I expect to see RuntimeWarnings in this block for mean of empty slice
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        if score_range["sorting"] == 1:  # low to high
            sort_order = np.argsort(np.nanmean(all_to_all, axis=0))
            cmap = "rocket_r"
        else:
            cmap = "rocket"
            sort_order = np.argsort(np.nanmean(all_to_all, axis=0))[::-1]

    all_to_all = all_to_all[:, sort_order]
    all_to_all = all_to_all[sort_order, :]
    segmentation_types = [segmentation_types[s] for s in sort_order]

    # write out csvs of data:
    output_directory = f"{output_path}/csvs/{group}/{crop}/{metric_value}"
    os.makedirs(output_directory, exist_ok=True)
    try:
        os.remove(f"{output_directory}/{organelle_name}.csv")
    except OSError:
        pass

    ignored = ["refinements", "ariadne"]
    with open(f"{output_directory}/{organelle_name}.csv", "w") as f:
        f.write(f"annotator 1,annotator 2,{metric_name}\n")
        for i in range(len(segmentation_types)):
            for j in range(i):
                if (
                    segmentation_types[i] not in ignored
                    and segmentation_types[j] not in ignored
                ):
                    f.write(
                        f"{segmentation_types[i]},{segmentation_types[j]},{all_to_all[i][j]}\n"
                    )

    for color_range in ["standard", "combined"]:
        _, ax = plt.subplots(1, 1, figsize=(8, 6),)
        if color_range == "standard":
            # I expect to see RuntimeWarnings in this block for all-nan slice
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message="All-NaN slice encountere"
                )
                mask = ~np.eye(all_to_all.shape[0], dtype=bool)
                score_range_min = np.nanmin(all_to_all[mask])
                score_range_max = np.nanmax(all_to_all[mask])
                output_directory = (
                    f"{output_path}/plots_standard_color/{group}/{crop}/{metric_value}"
                )
        else:
            score_range_min = score_range["min"]
            score_range_max = score_range["max"]
            output_directory = f"{output_path}/plots/{group}/{crop}/{metric_value}"

        seaborn.heatmap(
            all_to_all,
            annot=True,
            square=True,
            vmin=score_range_min,
            vmax=score_range_max,
            cmap=cmap,
        )
        if score_range["sorting"] == 1:  # low to high
            plt.gcf().axes[1].invert_yaxis()

        ax.set_title(organelle_name)
        ax.collections[0].colorbar.set_label(metric_name)
        ax.xaxis.tick_top()
        plt.xticks(
            [i + 0.5 for i in range(len(segmentation_types))],
            segmentation_types,
            rotation=45,
            ha="left",
        )
        plt.yticks(
            [i + 0.5 for i in range(len(segmentation_types))],
            segmentation_types,
            rotation=0,
        )

        os.makedirs(output_directory, exist_ok=True)
        try:
            os.remove(f"{output_directory}/{organelle_name}.png")
        except OSError:
            pass
        plt.savefig(f"{output_directory}/{organelle_name}.png", bbox_inches="tight")
        plt.close()


def plot_boxplots(all_to_all_info):
    import pandas as pd
    df = pd.DataFrame(columns = ['Score Type', 'Organelle', 'Annotator', 'Pair', 'Score'])
    dict_for_dataframe = {}
    i = 0

    for key, info in all_to_all_info.items():
        organelle = key[1]
        score_type = key[2]
        scores_matrix = info["scores_matrix"]
        segmentation_types = info["segmentation_types"]
        for idx_one,segmentation_type_one in enumerate(segmentation_types):
            for idx_two,segmentation_type_two in enumerate(segmentation_types):
                if idx_one != idx_two:
                    score = scores_matrix[idx_one][idx_two]
                    dict_for_dataframe[i] = {"Score Type": score_type, "Organelle": organelle, "Annotator": segmentation_type_one, "Pair": f"{segmentation_type_one}_{segmentation_type_two}", "Score": score}
                    dict_for_dataframe[i+1] = {"Score Type": score_type, "Organelle": organelle, "Annotator": segmentation_type_two, "Pair": f"{segmentation_type_one}_{segmentation_type_two}", "Score": score}
                    i += 2

    df = pd.DataFrame.from_dict(dict_for_dataframe, "index")


    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    # print(px.strip(df.loc[df.total_bill>0], x="time", y="total_bill", animation_group="total_bill",color_discrete_sequence = ["red"]).data)
    fig = go.Figure()
    annotators = df["Annotator"].unique()
    for annotator in annotators:
        fig = fig.add_traces(
            px.strip(
                df.loc[df["Annotator"] == annotator],
                x="Organelle",
                y="Score",
                hover_data=["Annotator", "Pair", "Organelle", "Score"],
                color_discrete_sequence=["red"],
                stripmode="overlay",
            ).data
        )  # color_discrete_sequence = ["blue"], stripmode='overlay')
    fig.update_traces(
        {
            "marker.color": "rgba(0, 0, 0,0)",
            "marker.size": 3,
            "marker.line": dict(color="rgba(241, 145, 155,1.0)", width=1),
        }
    )
    for annotator in annotators:
        fig = fig.add_traces(
            px.strip(
                df.loc[df["Annotator"] == annotator],
                x="Organelle",
                y="Score",
                hover_data=["Annotator", "Pair", "Organelle", "Score"],
                stripmode="overlay",
            ).data
        )  # color_discrete_sequence = ["blue"], stripmode='overlay')
    for idx, trace in enumerate(fig.select_traces()):
        if idx >= len(annotators):
            trace.visible = False
            trace.marker.update(
                {
                    "color": "rgba(0, 0, 0,1.0)",
                    "size": 3,
                    "line": dict(color="rgba(0, 0, 0,1.0)", width=1),
                }
            )
    fig.add_trace(
        go.Box(x=df["Organelle"], y=df["Score"], boxpoints=False, marker_color="red")
    )
    # https://stackoverflow.com/questions/68894919/how-to-set-the-values-of-args-and-args2-in-plotlys-buttons-in-updatemenus
    updatemenus = []
    button_layer_y = -0.2
    for idx, annotator in enumerate(annotators):
        if idx % 5 == 0:
            button_layer_x = 1
            button_layer_y -= 0.3

        button_layer_x -= 0.2
        updatemenus.append(
            {
                "buttons": [
                    {
                        "method": "update",
                        "label": f"Toggle {annotator}",
                        "args": [
                            # 1. updates to the traces
                            {"visible": False},
                            # 2. updates to the layout
                            {},
                            # 3. which traces are affected
                            [idx + len(annotators)],
                        ],
                        "args2": [
                            {"visible": True},
                            # 2. updates to the layout
                            {},
                            # 3. which traces are affected
                            [idx + len(annotators)],
                        ],
                    }
                ],
                "type": "buttons",
                "direction": "down",
                "showactive": True,
                "xanchor": "left",
                "x": button_layer_x,
                "y": button_layer_y,
                "yanchor": "top",
            }
        )

    fig.update_layout(
        updatemenus=updatemenus,
        title="tmp",
        xaxis_title="Organelle",
        yaxis_title=score_type,
        showlegend=False,
    )
    fig.write_html("mine.html")
