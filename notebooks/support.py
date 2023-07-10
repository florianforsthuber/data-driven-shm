"""Support module holding functions for jupyter notebook
Functions which can be imported and used in jupyter notebooks.
Makes writing and debugging easier. Declutters notebooks and keeps
main focus on visualizing results.
When working on jupyter notebook, this code can be adjusted on the side with dummy data in
if __name__ section.
Author: student k1256205@students.jku.at
Created: 23/08/2022
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from string import Template
from scipy.interpolate import griddata

####### Figure Settings #####
def init_figure(rows, cols, figure_width_cm, figure_height_cm=None, use_ratio=True, sharey=False):
    # set figure layout
    if figure_height_cm is None:  #equivalent to use_ratio=True
        fig_w_in = figure_width_cm / 2.54
        fig_h_in = figure_width_cm / 1.618 / 2.54
    else:
        fig_w_in = figure_width_cm / 2.54
        fig_h_in = figure_height_cm / 2.54

    tex_settings_thesis = {"text.usetex": True,
                           "font.family": "serif",
                           "font.serif": "Computer Modern Roman",
                           "axes.labelsize": 10,
                           "font.size": 10,
                           "legend.fontsize": 8,
                           "xtick.labelsize": 8,
                           "ytick.labelsize": 8,
                           "lines.linewidth": 1}
    plt.rcParams.update(tex_settings_thesis)

    # initialize axis
    fig, ax = plt.subplots(rows, cols, figsize=(fig_w_in, fig_h_in), sharey=sharey, squeeze=False)
    return fig, ax


def get_sensor_information():
    sensor_labels = {1: {"x": 195.0, "y": 352.5, "desc_y": "V_1_1", "desc_x": "H_1_1", "desc_xi": "VH_1_1"},
                          2: {"x": 250.0, "y": 352.5, "desc_y": "V_2_1", "desc_x": "H_1_2", "desc_xi": "VH_1_2"},
                          3: {"x": 305.0, "y": 352.5, "desc_y": "V_3_1", "desc_x": "H_1_3", "desc_xi": "VH_1_3"},
                          4: {"x": 195.0, "y": 297.5, "desc_y": "V_1_2", "desc_x": "H_2_1", "desc_xi": "VH_2_1"},
                          5: {"x": 250.0, "y": 297.5, "desc_y": "V_2_2", "desc_x": "H_2_2", "desc_xi": "VH_2_2"},
                          6: {"x": 305.0, "y": 297.5, "desc_y": "V_3_2", "desc_x": "H_2_3", "desc_xi": "VH_2_3"},
                          7: {"x": 195.0, "y": 242.5, "desc_y": "V_1_3", "desc_x": "H_3_1", "desc_xi": "VH_3_1"},
                          8: {"x": 250.0, "y": 242.5, "desc_y": "V_2_3", "desc_x": "H_3_2", "desc_xi": "VH_3_2"},
                          9: {"x": 305.0, "y": 242.5, "desc_y": "V_3_3", "desc_x": "H_3_3", "desc_xi": "VH_3_3"},
                          }

    direction_map = {'desc_y': ['V_1_1', 'V_2_1', 'V_3_1', 'V_1_2', 'V_2_2', 'V_3_2', 'V_1_3', 'V_2_3', 'V_3_3'],
                     'desc_x': ['H_1_1', 'H_1_2', 'H_1_3', 'H_2_1', 'H_2_2', 'H_2_3', 'H_3_1', 'H_3_2', 'H_3_3'],
                     'desc_xi': ['VH_1_1', 'VH_1_2', 'VH_1_3', 'VH_2_1', 'VH_2_2', 'VH_2_3', 'VH_3_1', 'VH_3_2',
                            'VH_3_3']}
    grid_position = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                     4: (1, 0), 5: (1, 1), 6: (1, 2),
                     7: (2, 0), 8: (2, 1), 9: (2, 2)
                     }

    notation_map = {"H_1_1": r"$S^{1}_{x}$", "H_1_2": r"$S^{2}_{x}$", "H_1_3": r"$S^{3}_{x}$",
                    "H_2_1": r"$S^{4}_{x}$", "H_2_2": r"$S^{5}_{x}$", "H_2_3": r"$S^{6}_{x}$",
                    "H_3_1": r"$S^{7}_{x}$", "H_3_2": r"$S^{8}_{x}$", "H_3_3": r"$S^{9}_{x}$",
                    "V_1_1": r"$S^{1}_{y}$", "V_2_1": r"$S^{2}_{y}$", "V_3_1": r"$S^{3}_{y}$",
                    "V_1_2": r"$S^{4}_{y}$", "V_2_2": r"$S^{5}_{y}$", "V_3_2": r"$S^{6}_{y}$",
                    "V_1_3": r"$S^{7}_{y}$", "V_2_3": r"$S^{8}_{y}$", "V_3_3": r"$S^{9}_{y}$",
                    "VH_1_1": r"$S^{1}_{\xi}$", "VH_1_2": r"$S^{2}_{\xi}$", "VH_1_3": r"$S^{3}_{\xi}$",
                    "VH_2_1": r"$S^{4}_{\xi}$", "VH_2_2": r"$S^{5}_{\xi}$", "VH_2_3": r"$S^{6}_{\xi}$",
                    "VH_3_1": r"$S^{7}_{\xi}$", "VH_3_2": r"$S^{8}_{\xi}$", "VH_3_3": r"$S^{9}_{\xi}$"
                    }

    return sensor_labels, direction_map, notation_map, grid_position

####### DATA ANALYSIS #######
def get_FullSTD_data(source: str, damage_state: str):
    p_base = Path('../data/raw')
    p_target = p_base / source / damage_state
    files = sorted(p_target.glob(f"**/FullSTD*.csv"))
    df = pd.concat(pd.read_csv(f) for f in files)
    return df


def get_VSSG_data(source: str, damage_state: str, load_case=None):
    if load_case is not None:
        p_base = Path('../data/raw')
        p_target = p_base / source / damage_state / f"LC{load_case}"
    else:
        p_base = Path('../data/raw')
        p_target = p_base / source / damage_state

    files = sorted(p_target.glob(f"**/VSSG*.csv"))
    df = pd.concat(pd.read_csv(f) for f in files)
    return df

def get_full_field_VSSG_data(source: str, damage_state: str, load_case=None):
    if load_case is not None:
        p_base = Path('../data/raw')
        p_target = p_base / source / damage_state / f"LC{load_case}"
    else:
        p_base = Path('../data/raw')
        p_target = p_base / source / damage_state

    files = sorted(p_target.glob(f"**/VSSG*.csv"))
    df = pd.concat(pd.read_csv(f) for f in files)

    if source == "generated":
        df = convert_loadcase_generated_data(df)
    return df

def get_damaged_VSSG_data(source: str):
    p_base = Path('../data/raw')
    p_target = p_base / source
    files = sorted(p_target.glob(f"DS*/**/VSSG*.csv"))
    df = pd.concat(pd.read_csv(f) for f in files)
    return df

def get_experimental_VSSG():
    p_base = Path('../data/raw/experimental')
    # get pristine data and extend to VSSG format
    df_H = pd.read_csv(p_base / "Traindata_grid_python_exp_H.csv", sep=";")
    # insert missing label columns
    df_H.insert(1, "source", "experimental")
    df_H.insert(1, "damage_state", "pristine")
    df_H.insert(1, "damage_label", 0)
    # rename "Loadcase" column
    df_H.rename(columns={"Loadcase": "loadcase"}, inplace=True)

    # get damaged data DS1 and extend to VSSG format
    df_DS1 = pd.read_csv(p_base / "Traindata_grid_python_exp_DS1.csv", sep=";")
    # insert missing label columns
    df_DS1.insert(1, "source", "experimental")
    df_DS1.insert(1, "r", 6.25)
    df_DS1.insert(1, "y", 32.5)
    df_DS1.insert(1, "x", 105)
    df_DS1.insert(1, "damage_state", "DS1")
    df_DS1.insert(1, "damage_label", 1)
    # rename "Loadcase" column
    df_DS1.rename(columns={"Loadcase": "loadcase"}, inplace=True)

    # get damaged data DS1 and extend to VSSG format
    df_DS2 = pd.read_csv(p_base / "Traindata_grid_python_exp_DS2.csv", sep=";")
    # insert missing label columns
    df_DS2.insert(1, "source", "experimental")
    df_DS2.insert(1, "r", 9.5)
    df_DS2.insert(1, "y", 32.5)
    df_DS2.insert(1, "x", 105)
    df_DS2.insert(1, "damage_state", "DS2")
    df_DS2.insert(1, "damage_label", 1)
    # rename "Loadcase" column
    df_DS2.rename(columns={"Loadcase": "loadcase"}, inplace=True)

    df = pd.concat([df_H, df_DS1, df_DS2], ignore_index=True)
    return df

def convert_loadcase_generated_data(df_generated):
    # necessary for full-field VSSG data -> old format
    # replace value in loadcase column
    df = df_generated.copy()
    df[["loadcase", "sample_idx"]] = df["loadcase"].apply(sep_LC)
    return df
    # add new column gen_sample_idx at index 1

def sep_LC(LC):
    new_cols = LC.split('-')
    return pd.Series(list(map(int, new_cols)))

def preprocess(df, pipeline):
    """
    Helper function to apply a sklearn preporcess (normalization) function to the VSSG data.
    Normalization function is passed via a pipeline.
    :param df: dataframe as it is read from .csv file
    :param pipeline: sklearn.pipeline object
    :return: dataframe with normalized data
    """
    # preprocess data of df
    # get list of column names of df
    feature_names = [col for col in df.columns if 'V_' in col or 'H_' in col or 'VH_' in col]
    df_copy = df.copy()
    pipeline.fit(df_copy[feature_names])
    X = pipeline.transform(df_copy[feature_names])
    # return a dataframe object
    df_copy[feature_names] = X
    return df_copy


def plot_single_LC_sensor_strain(df, hue, mark=None, fill_min_max=None,
                                 direction=None, save_figure=False,
                                 figure_width_cm=12, figure_height_cm=None,
                                 title=None, subtitle=False, fn_extension=None,
                                 legend_outside=False, sensor_alignment=None, subplots=False,
                                 no_legend=False,
                                 y_label_norm=False):
    df = df.copy()

    _, direction_map, notation_map, _ = get_sensor_information()
    # reorder x-direction to match sequence to strain isoline direction
    if sensor_alignment == "coils":
        # reorder features to match sensor coil direction (similar to FOS)
        # xi-direction: align diagonals --> Bad idea because relationship of 11 and 33 are lost
        direction_map = {'desc_y': ['V_1_1', 'V_1_2', 'V_1_3', 'V_2_1', 'V_2_2', 'V_2_3', 'V_3_1', 'V_3_2', 'V_3_3'],
                         'desc_x': ['H_1_1', 'H_1_2', 'H_1_3', 'H_2_1', 'H_2_2', 'H_2_3', 'H_3_1', 'H_3_2', 'H_3_3'],
                         'desc_xi': ['VH_1_1', 'VH_2_1', 'VH_3_1', 'VH_1_2', 'VH_2_2', 'VH_3_2', 'VH_1_3', 'VH_2_3',
                                     'VH_3_3']}
    elif sensor_alignment == "x_isoline":
        # reorder x-direction to match sequence to strain isoline direction
        # xi-direction: leave as is
        direction_map = {'desc_y': ['V_1_1', 'V_2_1', 'V_3_1', 'V_1_2', 'V_2_2', 'V_3_2', 'V_1_3', 'V_2_3', 'V_3_3'],
                         'desc_x': ['H_1_1', 'H_2_1', 'H_3_1', 'H_1_2', 'H_2_2', 'H_3_2', 'H_1_3', 'H_2_3', 'H_3_3'],
                         'desc_xi': ['VH_1_1', 'VH_2_1', 'VH_3_1', 'VH_1_2', 'VH_2_2', 'VH_3_2', 'VH_1_3', 'VH_2_3',
                                     'VH_3_3']}

    hue_map = {"loadcase": "Loadcase",
               "damage_state": "Damage state",
               "source": "Source"}
    subtitle_map = {"desc_x": r"$x$-direction",
                    "desc_y": r"$y$-direction",
                    "desc_xi": r"$\xi$-direction"}

    _, _, notation_map, _ = get_sensor_information()

    # assemble feature according to requested direction
    features = []
    if direction is None:
        direction = ["desc_x", "desc_y", "desc_xi"]

    marker = get_markers(df, mark)
    df = df.set_index(hue)

    if subplots:
        # seperate directions into aligned subplots
        fig, axs = init_figure(1, len(direction),
                               figure_width_cm=figure_width_cm,
                               figure_height_cm=figure_height_cm,
                               sharey=True)

        for idx, direct in enumerate(direction):
            # for each direction -> plot data
            df_data = df[direction_map[direct]].T
            ax = df_data.plot(ax=axs[0, idx], style=marker, markersize=3, linestyle=':')  # axs[0, idx] because squeeze=False in init_figure -> subplots always returns a 2D array for axis

            if fill_min_max is not None:
                # fill_min_max = {"label": (y_min, y_max)}
                for label, data_points in fill_min_max.items():
                    y_min, y_max = data_points
                    axs[0, idx].fill_between(range(df_data.shape[0]),
                                     y_min[direction_map[direct]],
                                     y_max[direction_map[direct]], alpha=0.2, label=label)
            # set xticklabels for individual subplot
            axs[0, idx].set_xticks(range(df_data.shape[0]))
            formatted_lables = [notation_map[x_label] for x_label in list(df_data.index)]
            #axs[idx].set_xticklabels(formatted_lables, rotation=-90)
            axs[0, idx].set_xticklabels(formatted_lables)
            axs[0, idx].minorticks_on()
            axs[0, idx].tick_params(axis='x', which='minor', bottom=False)  # Turns minor  x-ticks off
            axs[0, idx].grid(which="major", linewidth=0.5)
            axs[0, idx].grid(which="minor", axis="y", linewidth=0.3, linestyle=':')
            # remove legends from subplots
            legend = axs[0, idx].get_legend()
            legend.remove()
            # add title to subplot
            if subtitle:
                axs[0, idx].title.set_text(subtitle_map[direct])
            if y_label_norm:
                axs[0, idx].set_ylabel(r"$\varepsilon$ normalized")
            else:
                axs[0, idx].set_ylabel(r"$\varepsilon$ in $\mu$m/m")
    else:
        # show data in a single figure
        fig, axs = init_figure(1, 1, figure_width_cm=figure_width_cm,
                               figure_height_cm=figure_height_cm)
        axs = axs[0, 0]
        for each in direction:
            features.extend(direction_map[each])
        df = df[features].T
        axs = df.plot(ax=axs, style=marker, markersize=3, linestyle=':')
        axs.set_xticks(range(df.shape[0]))
        formatted_lables = [notation_map[x_label] for x_label in list(df.index)]
        axs.set_xticklabels(formatted_lables)
        axs.minorticks_on()
        axs.tick_params(axis='x', which='minor', bottom=False)  # turn off minor y-ticks
        axs.grid(which="major", linewidth=0.5)
        axs.grid(which="minor", axis="y", linewidth=0.3, linestyle=':')
        legend = axs.get_legend()
        if y_label_norm:
            axs.set_ylabel(r"$\varepsilon$ normalized")
        else:
            axs.set_ylabel(r"$\varepsilon$ in $\mu$m/m")
    # x-axis: all 27 features
    # y-axis: strain value (feature value)

    if no_legend:
        legend.remove()
    elif legend_outside or subplots:
        legend.remove()
        handles = legend.legendHandles
        labels_list = [txt.get_text() for txt in legend.texts]
        fig.legend(handles, labels_list, title=hue_map[hue], alignment="left",
                   ncol=len(labels_list), loc="upper left",
                   bbox_to_anchor=(0.05, 1.13), fancybox=True)
        #axs.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    plt.show()

    if save_figure:
        target_dir = Path("../reports/figures/SingleLC_SensorStrain")
        filename = f"SingleLC_strain_{hue}"
        if fn_extension is not None:
            filename += f"_{fn_extension}"

        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')

def get_markers(df, mark):
    markers = ["o", "^", "s", "D", "v"]
    if mark is None:
        marker_seq = "o"
    else:
        unique = df[mark].unique()
        while len(unique) > len(markers):
            markers += markers[1:]
        marker_seq = df[mark].replace(unique, markers[:len(unique)]).to_list()
    return marker_seq

def sns_plot_single_LC_sensor_strain(df, hue, mark=None, direction=None,
                                     alpha=1.0, linewidth=0.7, markersize=4,
                                     save_figure=False,
                                     figure_width_cm=12, figure_height_cm=None, fn_extension=None,
                                     xlabel="Feature", ylabel="Value",
                                     fill_min_max=None,):
    sensor_labels, direction_map, notation_map, grid_position = get_sensor_information()
    hue_map = {"loadcase": "Loadcase",
              "damage_state": "Damage state",
              "source": "Source"}

    if direction is None:
        direction = ["desc_x", "desc_y", "desc_xi"]
    features = []
    for each in direction:
        features.extend(direction_map[each])

    # melt DataFrame
    df_data = df.copy()
    df_data.index = df_data.index.set_names(["sample"])
    df_data = df_data.reset_index()

    cols = df_data.columns
    cols_to_keep = [each for each in cols if each not in features]
    cols_to_values = features

    df_melt = df_data.melt(id_vars=cols_to_keep, value_vars=cols_to_values, var_name="feature", value_name="value")

    # seperate directions into aligned subplots
    fig, axs = init_figure(1, len(direction),
                           figure_width_cm=figure_width_cm,
                           figure_height_cm=figure_height_cm,
                           sharey=True)
    # handle hue, marker combinations
    unique_hue_elems = len(df_data[hue].unique())
    color_palette = sns.color_palette("deep", n_colors=unique_hue_elems)
    for idx, direct in enumerate(direction):
        # for each direction -> plot data
        df_subplot = df_melt[df_melt["feature"].isin(direction_map[direct])]
        g = sns.lineplot(ax=axs[0, idx], data=df_subplot, x="feature", y="value", hue=hue,
                         style=mark, units="sample", estimator=None,
                         alpha=alpha, markers=True, dashes=False,
                    linestyle=':', linewidth=linewidth, markersize=markersize,
                    palette=color_palette, markeredgecolor="none")
        # set xticklabels for individual subplot
        #axs[0, idx].set_xticks(range(df_subplot.shape[0]))
        formatted_lables = [notation_map[x_label.get_text()] for x_label in axs[0, idx].get_xticklabels()]
        axs[0, idx].set_xticklabels(formatted_lables)
        # handle grid and minor ticks
        axs[0, idx].minorticks_on()
        axs[0, idx].tick_params(axis='x', which='minor', bottom=False)  # Turns minor  x-ticks off
        axs[0, idx].grid(which="major", linewidth=0.5)
        axs[0, idx].grid(which="minor", axis="y", linewidth=0.3, linestyle=':')

        # show range of data via fill (intended to show range of numerical samples)
        # fill_min_max = {"min": Series, "max": Series}
        if fill_min_max is not None:
            axs[0, idx].fill_between(range(len(direction_map[direct])),
                                     fill_min_max["min"][direction_map[direct]],
                                     fill_min_max["max"][direction_map[direct]],
                                     alpha=0.3,
                                     color="lightgray",
                                     edgecolor="dimgray")

        # remove legends from subplots
        legend = axs[0, idx].get_legend()
        handles, labels = axs[0, idx].get_legend_handles_labels()
        if idx < len(direction) - 1:
            legend.remove()
        # set custom x-axis label
        axs[0, idx].set_xlabel(xlabel)
    # set custom y-axis label
    axs[0, 0].set_ylabel(ylabel)
    # move legend outside
    legend = axs[0, idx].legend()
    for text in legend.texts:
        if text.get_text() == hue:
            text.set_text(hue_map[hue])
        elif text.get_text() == mark:
            text.set_text(hue_map[mark])
    sns.move_legend(axs[0, idx], "upper left", bbox_to_anchor=(1, 1))

    plt.show()

    if save_figure:
        target_dir = Path("../reports/figures/SingleLC_SensorStrain")
        filename = f"SingleLC_strain_{hue}"
        if fn_extension is not None:
            filename += f"_{fn_extension}"

        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')

def plot_sensor_grid_histograms(df, hue, direction, hist_params=None,
                                y_lim_max=150, bins=20, equal_bin_axis=False,
                                add_vert_line_OLC_data=False, add_vert_line_gen_sample=None,
                                save_figure=False, fn_extension=None, title=None):
    df = df.copy()
    feature_names = [col for col in df.columns if 'V_' in col or 'H_' in col or 'VH_' in col]

    fig, axs = init_figure(3, 3, figure_width_cm=15, figure_height_cm=15, use_ratio=False, sharey=True)
    if hist_params is None:
        hist_params = {'bins': bins, 'stat': 'count', 'kde': True, 'binrange': None}

    sensor_labels, direction_map, notation_map, grid_position = get_sensor_information()

    #find min max data values of given direction

    xmax = df[direction_map[direction]].to_numpy().max()
    xmin = df[direction_map[direction]].to_numpy().min()

    xmin -= abs(xmin) * 0.2
    xmax += abs(xmax) * 0.2

    s = Template('\\texttt{${label}}')
    for sensor_idx, (i, j) in grid_position.items():
        x_label = sensor_labels[sensor_idx][direction]
        #x_label_formatted = s.substitute(label=x_label)
        x_label_formatted = notation_map[x_label]
        ax_sns = sns.histplot(df[feature_names],
                     x=sensor_labels[sensor_idx][direction],
                     hue=df[hue],
                     ax=axs[i, j],
                     bins=hist_params['bins'],
                     stat=hist_params['stat'],
                     kde=hist_params['kde'],
                     binrange=hist_params['binrange']
                     )
        #ax_sns.set_title(f"Sensor {sensor_idx}")
        ax_sns.set_xlabel(x_label_formatted)
        if y_lim_max is not None:
            ax_sns.set_ylim(0, y_lim_max)
        if equal_bin_axis:
            ax_sns.set_xlim(xmin, xmax)  # Add 20% margin to sides
        #ax_sns.grid(linewidth=0.5)
        ax_sns.minorticks_on()
        ax_sns.tick_params(axis='y', which='minor', left=False)
        ax_sns.grid(which="major", linewidth=0.5)
        ax_sns.grid(which="minor", axis="x", linewidth=0.3, linestyle=':')
        legend = ax_sns.get_legend()
        legend.remove()

    # TODO: adapt get stats for experimental data (two different damage states!)
    #  or add parameter to choose damage states ?
    df_stats = get_stats(df, hue, direction, direction_map, round=True)
    if add_vert_line_OLC_data:
        objective_loadcase = df.loc[df["source"] == "generated", "loadcase"].unique()[0]
        for sensor_idx, (i, j) in grid_position.items():
            current_x = sensor_labels[sensor_idx][direction]
            x_vert = df.loc[(df["source"] == "simulated") & (df["loadcase"] == objective_loadcase), current_x].values[0]
            axs[i, j].axvline(x=x_vert, color="blue", alpha=0.5, ls="--", lw=1.0)
        pass #TODO: add mean of generated data and value of specific loadcase

    if add_vert_line_gen_sample is not None:
        for sensor_idx, (i, j) in grid_position.items():
            current_x = sensor_labels[sensor_idx][direction]
            x_vert = df.loc[(df["source"] == "generated") & (df["loadcase"] == objective_loadcase) & (df["sample_idx"] == float(add_vert_line_gen_sample)), current_x].values[0]
            axs[i, j].axvline(x=x_vert, color="red", alpha=0.5, ls="-.", lw=1.0)

    handles = legend.legendHandles
    labels_list = [txt.get_text() for txt in legend.texts]
    fig.legend(handles, labels_list, ncol=len(labels_list), loc="upper left", bbox_to_anchor=(0, 1.03))
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    plt.show()

    if save_figure:
        target_dir = Path("../reports/figures/Histograms_SensorGrid")
        filename = f"Histogram_SensorGrid_{hue}_{direction}_bins{bins}"
        if equal_bin_axis:
            filename += f"_equBinAxis"
        if add_vert_line_OLC_data:
            filename += f"_OLCindicator"
        if fn_extension is not None:
            filename += f"_{fn_extension}"

        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')
    return df_stats

def plot_delta_heat_map(df_sample, direction=None, show_damage=False, show_sensors=True, custom_axis_scaling=None,
                        cbar_label=r"prediction error $\delta$",
                        figure_width_cm=15, figure_height_cm=None, save_figure=False, fn_extension=None):
    # direction: pass list of directions -> sum up sensors of given directions

    df = df_sample.copy()
    source = df["source"].values[0]
    damage_state = df["damage_state"].values[0]
    loadcase = df["loadcase"].values[0]

    sensor_labels, direction_map, notation_map, grid_position = get_sensor_information()
    df_sensor_labels = pd.DataFrame.from_dict(sensor_labels.values())

    # local to global coord. sys. offset
    delta_x = 140
    delta_y = 187.5

    #points = (df_sensor_labels["x"].to_numpy(), df_sensor_labels["y"].to_numpy())

    df_points = df_sensor_labels[["x", "y"]].to_numpy()


    # mesh size of interpolation grid
    x_step = y_step = 1

    x_max = 360.0
    y_max = 380.0
    x_min = 140.0
    y_min = 187.5

    # get edge nodes with known 0 delta
    x_edge = np.linspace(x_min, x_max, 5)
    y_edge = np.linspace(y_min, y_max, 5)
    # assemble edge coordinates
    x_points = np.concatenate((df_sensor_labels["x"].to_numpy(),
                               x_edge, x_edge,
                               np.full(len(y_edge), 140.0), np.full(len(y_edge), 360.0)))

    y_points = np.concatenate((df_sensor_labels["y"].to_numpy(),
                               np.full(len(x_edge), 187.5), np.full(len(y_edge), 380.0),
                               y_edge, y_edge))

    points = (x_points, y_points)

    # interpolation grid
    x = np.linspace(x_min, x_max, int((x_max - x_min) * (1 / x_step) + 1))
    y = np.linspace(y_min, y_max, int((y_max - y_min) * (1 / y_step) + 1))
    x_mat, y_mat = np.meshgrid(x, y)
    grid = (x_mat, y_mat)
    #grid = np.meshgrid(x, y)

    # get deltas of each feature
    interpolations = []
    for direct in direction:
        features = direction_map[direct]
        deltas = df[features].to_numpy()[0] #assuming they have the same order as given in sensor labels
        values = np.concatenate((deltas, np.zeros(len(x_edge) * 4)))

        # get interpolation
        delta_interpolation = griddata(points, values, grid, method="cubic")
        delta_interpolation = np.where(delta_interpolation < 0, 0, delta_interpolation)  # eliminate negative values resulting from cubic interpolation
        interpolations.append(delta_interpolation)
        # stack delta_interpolation in new axis and sum at the end
    summed_interpolations = np.sum(np.stack(interpolations, axis=0), axis=0)

    fig, axs = init_figure(1, 1,
                           figure_width_cm=figure_width_cm,
                           figure_height_cm=figure_height_cm,
                           sharey=True)
    ax = axs[0, 0]

    # contour plot setup
    if custom_axis_scaling is not None:
        # find min/max values of provided data beforehand and pass them on for equal axis scaling of each iterand
        min_value, max_value, n_steps = custom_axis_scaling
        contour_levels = list(range(min_value, max_value, n_steps))
    else:
        contour_levels = 20

    cmap = "viridis" # "viridis_r"

    cs = ax.contourf(x_mat, y_mat, summed_interpolations,
                     levels=contour_levels,
                     #vmax=200,
                     cmap=cmap)

    # preserve aspect ratio
    ax.set_aspect("equal")
    # add custom axis for colorbar
    cbar_width = 0.02
    cax = fig.add_axes([ax.get_position().x1 + 0.02,
                        ax.get_position().y0,
                        cbar_width,
                        ax.get_position().height
                        ])

    # add colorbar
    cbar = fig.colorbar(cs, cax=cax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(r'${}_{I}x$ in mm')
    ax.set_ylabel(r'${}_{I}y$ in mm')

    if show_sensors:
        for idx, sensor in sensor_labels.items():
            ax.plot(sensor["x"], sensor["y"], ".", color="black")

    if show_damage:
        file_path = Path.cwd().parent / "models" / "FEM_model" / "scripts" / "Damage_locations.csv"

        df_damages = pd.read_csv(file_path)

        # handle experimental damage location exception
        if source == "experimental":
            df_damages_exp = pd.DataFrame([{"name": "DS1", "x": 105, "y": 32.5, "radius": 6.25},
                                          {"name": "DS2", "x": 105, "y": 32.5, "radius": 9.5}])
            damage_location = df_damages_exp.loc[df_damages_exp["name"] == damage_state]
        else:
            damage_location = df_damages.loc[df_damages["name"] == damage_state]

        #ax = fig.gca()
        pos_x = damage_location['x'].values[0] + delta_x  # convert damage coords to global coords
        pos_y = damage_location['y'].values[0] + delta_y  # convert damage coords to global coords
        radius = damage_location['radius'].values[0]

        ax.add_patch(patches.Circle((pos_x, pos_y), radius,
                                    facecolor="none",
                                    edgecolor="red",
                                    linewidth=1,
                                    linestyle="--"))


    plt.show()
    # save figure
    direction_str = ""
    for each in direction:
        direction_str += f"{each}_"
    if save_figure:
        target_dir = Path("../reports/figures/Delta_heat_map")
        filename = f"Delta_heat_map_{direction_str}{damage_state}_{loadcase}_{source}_w{figure_width_cm}cm"
        if fn_extension is not None:
            filename += f"_{fn_extension}"

        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')


def scatter_damage_index(damage_index, y_test, threshold, benchmark, n=None, xticks=True,
                         xlabel=r"Evaluated sample",
                         ylabel=r"Damage index",
                         figure_width_cm=12, figure_height_cm=None, save_figure=False, fn_extension=""):
    # reconnect damage_index and y_test
    df_damage_index = pd.DataFrame({"damage_index": damage_index})
    df = pd.concat([df_damage_index, y_test], axis=1)
    # reduce size by sampling if requested
    if n is not None:
        df = df.sample(n=n).reset_index(drop=True)

    # init figure
    fig, axs = init_figure(1, 1,
                           figure_width_cm=figure_width_cm,
                           figure_height_cm=figure_height_cm,
                           sharey=True)
    ax = axs[0, 0]
    labels = list(df["damage_state"].replace(regex={r"^DS.*$": "damaged"}))

    ax = sns.scatterplot(ax=ax, x=df.index, y=df["damage_index"], hue=labels, alpha=0.7)
    ax.axhline(y=threshold, color='r', linestyle='-', label="threshold")
    ax.axhline(y=benchmark, color='g', linestyle='-', label="benchmark")

    if xticks:
        ax.set_xticks(range(len(df.index)))
        ax.minorticks_on()

    #ax.minorticks_on()
    #ax.tick_params(axis='x', which='minor', bottom=False)  # Turns minor  x-ticks off
    #ax.grid(which="major", linewidth=0.5)
    ax.grid("on")

    # grid linewidth?
    # label for decision lines?
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.show()

    if save_figure:
        target_dir = Path("../reports/figures/Damage_indices")
        filename = f"Damage_indices_w{figure_width_cm}cm"
        if fn_extension is not None:
            filename += f"_{fn_extension}"
        figure_path = target_dir / filename
        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')
    pass

def plot_embedding(X_transformed, damage_states, figure_width_cm=12,
                   figure_height_cm=None,
                   xlabel=r"First MDS coordinate",
                   ylabel=r"Second MDS coordinate",
                   save_figure=False, fn_extension=""):
    damage_states_num = damage_states
    for idx, each in enumerate(damage_states_num.unique()):
        damage_states_num = damage_states_num.replace(each, idx)

    # init figure
    fig, axs = init_figure(1, 1,
                           figure_width_cm=figure_width_cm,
                           figure_height_cm=figure_height_cm,
                           sharey=True)
    ax = axs[0, 0]
    ax = sns.scatterplot(ax=ax, x=X_transformed[:, 0], y=X_transformed[:, 1], hue=damage_states.values, alpha=0.7)
    #scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=damage_states.values)
    #plt.legend(handles=ax.legend_elements()[0], labels=list(damage_states.unique()))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()
    if save_figure:
        target_dir = Path("../reports/figures/Embedding")
        filename = f"Embedding_w{figure_width_cm}cm"
        if fn_extension is not None:
            filename += f"_{fn_extension}"
        figure_path = target_dir / filename
        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')



def get_stats(df, hue, direction, direction_map, round=True):
    if hue == "source":
        assert hue == "source", "Only hue=source implemented."
        # mean of each feature column
        mean_sim = df[direction_map[direction]].loc[df[hue] == "simulated"].mean()
        # stamdard deviation of each feature column
        std_sim = df[direction_map[direction]].loc[df[hue] == "simulated"].std(ddof=0) #To have the same behaviour as numpy.std, use ddof=0 (instead of the default ddof=1)

        # mean of each feature column
        mean_gen = df[direction_map[direction]].loc[df[hue] == "generated"].mean()
        # stamdard deviation of each feature column
        std_gen = df[direction_map[direction]].loc[df[hue] == "generated"].std(ddof=0)

        # calculate ratios
        delta_mean = mean_sim - mean_gen
        ratio_std = std_gen / std_sim

        # calculate divergence
        #div_mean = (mean_sim - mean_gen)/mean_sim * 100
        div_std = (ratio_std - 1) * 100

        df_stats = pd.concat([mean_sim.rename("mean_sim"),
                              std_sim.rename("std_sim"),
                              mean_gen.rename("mean_gen"),
                              std_gen.rename("std_gen"),
                              delta_mean.rename("delta_mean"),
                              ratio_std.rename("ratio_std"),
                              div_std.rename("div_std")], axis=1)

    elif hue == "damage_state":
        stats_list = []
        for damage_state in df["damage_state"].unique():
            # mean of specific damage state
            mean = df[direction_map[direction]].loc[df[hue] == damage_state].mean()
            stats_list.append(mean.rename(f"mean_{damage_state}"))
            # standard deviation of specific damage state
            std = df[direction_map[direction]].loc[df[hue] == damage_state].std(ddof=0)
            stats_list.append(std.rename(f"std_{damage_state}"))
        df_stats = pd.concat(stats_list, axis=1)
    else:
        # return empty dataframe
        df_stats = pd.DataFrame()
    if round:
        df_stats = df_stats.round(2)

    return df_stats

if __name__ == '__main__':
    pass