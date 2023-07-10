"""Support file for damage detection notebooks
Author: student k1256205@students.jku.at
Created: 17/03/2023
"""
# import general modules
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# import specific functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import matplotlib.patches as patches
import matplotlib.patches as mpatches

from support import get_VSSG_data
from support import get_experimental_VSSG
from support import get_damaged_VSSG_data
from support import init_figure
from support import plot_single_LC_sensor_strain
from support import get_sensor_information
from support import get_markers

# implement modular sklearn pipeline

def prepare_raw_data(df, direction=["desc_x", "desc_y", "desc_xi"]):
    direction_map = {'desc_y': ['V_1_1', 'V_2_1', 'V_3_1', 'V_1_2', 'V_2_2', 'V_3_2', 'V_1_3', 'V_2_3', 'V_3_3'],
                     'desc_x': ['H_1_1', 'H_1_2', 'H_1_3', 'H_2_1', 'H_2_2', 'H_2_3', 'H_3_1', 'H_3_2', 'H_3_3'],
                     'desc_xi': ['VH_1_1', 'VH_1_2', 'VH_1_3', 'VH_2_1', 'VH_2_2', 'VH_2_3', 'VH_3_1', 'VH_3_2',
                                 'VH_3_3']}
    # assemble feature according to requested direction
    features = []
    if direction is None:
        direction = ["desc_x", "desc_y", "desc_xi"]
    for each in direction:
        features.extend(direction_map[each])
    # make loadcase numbers new index
    df = df.set_index("loadcase")
    # data subsets containing features and lables
    X_data = df[features]
    y_data = df.loc[:, ("damage_label", "damage_state", "source")]
    y_data["damage_label"] = y_data["damage_label"].replace({0: 1, 1: -1})

    return X_data, y_data

def split_data(df, direction=None, test_size=0.3, random_state=0):
    """
    :param df: dataframe containing VSSG dataset to be split
    :param direction: list of directions to isolate: ["desc_x", "desc_y", "desc_xi"], None: Use all directions
    :param test_size: float to give percent or int to give total amount
    :return:
    """
    # adapt label format to sklearn standard, optionally isolate direction
    X, y = prepare_raw_data(df, direction=direction)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def assemble_set1_OC(df_pristine, df_damaged,
                     direction=None,
                     train_size=0.8,
                     test_size_damaged=200,
                     balance_test_set=True,
                     random_seed=0):
    """
    Dataset assembly for training with simulated data, testing with simulated data
    :param df_pristine: dataframe pristine data
    :param df_damaged: dataframe damaged data
    :param direction: direction isolation -> list
    :param train_size: size of training set in percent
    :param test_size_damaged: absolute size of damaged samples in test set if balance_test_set is False
    :param balance_test_set: equal amount of damaged and pristine data in test set
    :return: Data and labels
    """
    # do pristine split
    test_size = round(1 - train_size, 3)  # flatten out numerical uncertainty
    X_train, X_test_pristine, y_train, y_test_pristine = split_data(df_pristine, direction,
                                                                    test_size, random_state=random_seed)

    # do damage split
    if balance_test_set:
        _, X_test_damaged, _, y_test_damaged = split_data(df_damaged, direction,
                                                          test_size=X_test_pristine.shape[0], random_state=random_seed)
    else:
        _, X_test_damaged, _, y_test_damaged = split_data(df_damaged, direction,
                                                          test_size=test_size_damaged, random_state=random_seed)

    # assemble test set
    X_test = pd.concat([X_test_pristine, X_test_damaged], ignore_index=False)
    y_test = pd.concat([y_test_pristine, y_test_damaged], ignore_index=False)

    return X_train, X_test, y_train, y_test


def assemble_set2_OC(df_simulated_pristine, df_simulated_damaged, df_generated_pristine,
                     direction=None,
                     train_size=0.8,
                     test_size_damaged=200,
                     balance_test_set=True,
                     random_seed=0):
    """
    !! Due to the way the generated data was derived from simulated data, there is the chance of creating a data leak,
    when creating training and test set including generated and simulated data.
    Those generated samples derived from a loadcase which matches a loadcase included in the test set (simulated)
    needs to be excluded.

    :param df_simulated_pristine:
    :param df_simulated_damaged:
    :param df_generated_pristine:
    :param direction:
    :param train_size:
    :param test_size_damaged:
    :param balance_test_set:
    :param random_seed:
    :return:
    """

    # split training data -> generated pristine
    test_size = round(1 - train_size, 3)  # flatten out numerical uncertainty
    X_train, _, y_train, _ = split_data(df_generated_pristine, direction, test_size, random_state=random_seed)

    # assemble test data -> simulated pristine/damaged --> order is not relevant for test data (?)
    if balance_test_set:
        df = pd.concat([df_simulated_pristine.sample(n=test_size_damaged, random_state=random_seed),
                        df_simulated_damaged.sample(n=test_size_damaged, random_state=random_seed)],
                       ignore_index=True)
    else:
        df = pd.concat([df_simulated_pristine,
                        df_simulated_damaged.sample(n=test_size_damaged, random_state=random_seed)],
                       ignore_index=True)

    X_test, y_test = prepare_raw_data(df, direction=direction)

    return X_train, X_test, y_train, y_test


def assemble_set3_OC(df_simulated_pristine, df_simulated_damaged, df_generated_pristine,
                     direction=None,
                     train_size_simulated=0.8,
                     train_size_total=10000,
                     test_size_damaged=200,
                     balance_test_set=True,
                     random_seed=0):


    # split simulated pristine data
    test_size = round(1 - train_size_simulated, 3)
    df_simulated_pristine_train, df_simulated_pristine_test = train_test_split(df_simulated_pristine,
                                                                               test_size=test_size, random_state=random_seed)
    # assemble train data
    gen_data_size = train_size_total - df_simulated_pristine_train.shape[0] + 1  # +1 so that full train set can be selected through train_test_split function
    df_train = pd.concat([df_simulated_pristine_train,
                          df_generated_pristine.sample(n=gen_data_size, random_state=random_seed)])
    X, y = prepare_raw_data(df_train, direction=direction)
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size_total, random_state=random_seed)

    # assemble test set
    if balance_test_set:
        df = pd.concat([df_simulated_pristine_test,
                        df_simulated_damaged.sample(n=df_simulated_pristine_test.shape[0], random_state=random_seed)],
                       ignore_index=True)
    else:
        df = pd.concat([df_simulated_pristine_test,
                        df_simulated_damaged.sample(n=test_size_damaged, random_state=random_seed)],
                       ignore_index=True)
    X_test, y_test = prepare_raw_data(df, direction=direction)

    return X_train, X_test, y_train, y_test


def assemble_set4_OC(df_simulated_pristine, df_simulated_damaged, df_generated_pristine,
                     direction=None,
                     train_size=0.8,
                     test_size_damaged=200,
                     balance_test_set=True,
                     random_seed=0):
    """
    !! Due to the way the generated data was derived from simulated data, there is the chance of creating a data leak,
    when creating training and test set including generated and simulated data.
    Those generated samples derived from a loadcase which matches a loadcase included in the test set (simulated)
    needs to be excluded.

    :param df_simulated_pristine:
    :param df_simulated_damaged:
    :param df_generated_pristine:
    :param direction:
    :param train_size:
    :param test_size_damaged:
    :param balance_test_set:
    :param random_seed:
    :return:
    """

    # 1. create testset from simulated data
    # assemble test data -> simulated pristine/damaged --> order is not relevant for test data
    if balance_test_set:
        df = pd.concat([df_simulated_pristine.sample(n=test_size_damaged, random_state=random_seed),
                        df_simulated_damaged.sample(n=test_size_damaged, random_state=random_seed)],
                       ignore_index=True)
    else:
        df = pd.concat([df_simulated_pristine,
                        df_simulated_damaged.sample(n=test_size_damaged, random_state=random_seed)],
                       ignore_index=True)

    X_test, y_test = prepare_raw_data(df, direction=direction)

    # 2. drop samples from generated data which match loadcase number present in testset
    df_gen = df_generated_pristine.copy()
    testset_OLCs = X_test.index.unique().to_list()
    df_gen_filtered = df_gen[~df_gen["loadcase"].isin(testset_OLCs)]

    # 3. assemble training set from filtered generated data
    # split training data -> generated pristine
    test_size = round(1 - train_size, 3)  # flatten out numerical uncertainty
    X_train, _, y_train, _ = split_data(df_gen_filtered, direction, test_size, random_state=random_seed)

    return X_train, X_test, y_train, y_test

def find_similar_samples(df, df_val_set, nr_select):
    """
    Function to find samples which are most similar to experimental samples (validation set).
    Similarity here is defined as the mean of absolut feature value differences from experimental data.
    :return: nr_select most similar samples of df
    """
    # copy df
    df = df.copy().reset_index(drop=True)
    df_val_set = df_val_set.copy()
    features = [col for col in df.columns if 'V_' in col or 'H_' in col or 'VH_' in col]
    # calculate mean of df_val_set
    df_val_mean = df_val_set[features].mean()
    # add score column -> only return those rows with
    features = [col for col in df.columns if 'V_' in col or 'H_' in col or 'VH_' in col]
    df_delta = df[features].sub(df_val_mean[features], axis="columns").abs()
    score = df_delta.mean(axis=1)
    df_most_similar = df.iloc[score.sort_values()[:nr_select].index]
    return df_most_similar

def remove_samples(df_train, df_test):
    """
    Function to remove loadcases from training data which already appear in test data.
    This determined via a common objective loadcase
    :return:
    """
    df_train = df_train.copy()
    testset_OLCs = df_test["loadcase"].unique()
    return df_train[~df_train["loadcase"].isin(testset_OLCs)]

def remove_n_olcs(df, n_to_remove, random_seed=0):
    """
    Function removes n_to_remove randomly picked objective loadcases from the generated data set df.
    :param df: generated VSSG data
    :param n_to_remove: amount  of random OLCs to remove
    :return: thinned out dataset
    """
    df_gen = df.copy()
    rng = np.random.default_rng(random_seed)
    current_olcs = df_gen["loadcase"].unique()
    remove_olcs = rng.choice(current_olcs, n_to_remove, replace=False)
    return df_gen[~df_gen["loadcase"].isin(remove_olcs)]

def get_feature_deltas(X_data: pd.DataFrame):
    """
    Maybe use this as apply function on X.
    Feature construction by simply calculation the difference between two neighbouring features.
    The individual diffrences respect the spatial relationship of the sensors (which is known).
    :param X: raw data in form of a dataframe
    :return: new data X with constructed features
    """
    # define the delta mapping between standard features and delta-features
    # order of the features in provided data should be correct -> reflect the sensor numbering in correct sequence
    # check if all directions are used
    new_feature_names = ["DX_12", "DX_23", "DX_45", "DX_56", "DX_78", "DX_89",
                         "DX_14", "DX_47", "DX_25", "DX_58", "DX_36", "DX_69",
                         "DY_12", "DY_23", "DY_45", "DY_56", "DY_78", "DY_89",
                         "DY_14", "DY_47", "DY_25", "DY_58", "DY_36", "DY_69",
                         "DXi_12", "DXi_23", "DXi_45", "DXi_56", "DXi_78", "DXi_89",
                         "DXi_14", "DXi_47", "DXi_25", "DXi_58", "DXi_36", "DXi_69",
                         ]
    X_diff = pd.DataFrame()
    X_diff[new_feature_names] = X_data.apply(convolutions, axis=1)
    return X_diff

def convolutions(X_sample):
    x_mat = X_sample.to_numpy().reshape(-1, 3, 3) #this could also work
    x_mat_T = np.transpose(x_mat, axes=(0, 2, 1))

    # horizontal convolution
    diff_H = convolute_neighbor(x_mat)

    # vertical convolution
    diff_V = convolute_neighbor(x_mat_T)

    # combine and flatten results
    diff_features = np.concatenate((diff_H, diff_V), axis=1).flatten()
    return pd.Series(diff_features)


def convolute_neighbor(X, kernel=None):
    """
    :param x_mat: sample in matrix format, shape (direction, n_row, n_col)
    :param kernel:
    :return:
    """
    if kernel is None:
        kernel = np.array([-1, 1])  # actually [1, -1] but convolve flips the sliding kernel
    # use numpy convolve function
    n_dir, n_row, n_col = X.shape
    out = []
    for direction in range(n_dir):
        out_elems = []
        for row in range(n_row):
            x_vec = X[direction][row]
            out_elems.extend(np.convolve(x_vec, kernel, mode="valid"))
        out.append(out_elems)
    return np.stack(out) #shape (n_dir, 6)


def apply_classifiers(X_train, X_test, y_train, y_test, classifiers: dict):
    clf_results = {}
    clfs = {}
    for name, pipeline in classifiers.items():
        clf = pipeline.fit(X_train, y_train["damage_label"])
        clfs[name] = clf
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.decision_function(X_test)
        clf_results[name] = {"y_true": y_test, "y_pred": y_pred, "y_score": y_score}
    return clfs, clf_results


def check_clf_consistency(default_set,
                          default_direction,
                          random_seed_runs,
                          df_simulated_pristine,
                          df_simulated_damaged,
                          df_generated_pristine,
                          test_size_damaged,
                          classifiers):
    clf_metrics_lst = []
    for random_seed in range(random_seed_runs):
        #clfs = classifiers
        clfs = {"IF": Pipeline([('scaler', StandardScaler()),
                                         ('pca', PCA(n_components=9)),
                                         ('IF', IsolationForest(n_estimators=150, random_state=0))]),
                        "OC-SVM": Pipeline([('scaler', StandardScaler()),
                                            ('pca', PCA(n_components=9)),
                                            ('OC-SVM', OneClassSVM())]),
                        "Elliptic-Env": Pipeline([('scaler', StandardScaler()),
                                                  ('pca', PCA(n_components=9)),
                                                  ('Elliptic-Env', EllipticEnvelope(random_state=0))]),
                        "LOF": Pipeline([('scaler', StandardScaler()),
                                         ('pca', PCA(n_components=9)),
                                         ('LOF', LocalOutlierFactor(n_neighbors=20, novelty=True))])}

        # assemble requested data set
        if default_set == "set1":
            X_train_set, X_test_set, y_train_set, y_test_set = assemble_set1_OC(df_simulated_pristine,
                                                                                            df_simulated_damaged,
                                                                                            direction=default_direction,
                                                                                            balance_test_set=True,
                                                                                            random_seed=random_seed)
        elif default_set == "set2":
            X_train_set, X_test_set, y_train_set, y_test_set = assemble_set2_OC(df_simulated_pristine,
                                                                                            df_simulated_damaged,
                                                                                            df_generated_pristine,
                                                                                            direction=default_direction,
                                                                                            test_size_damaged=200,
                                                                                            balance_test_set=True,
                                                                                            random_seed=random_seed)
        elif default_set == "set3":
            X_train_set, X_test_set, y_train_set, y_test_set = assemble_set3_OC(df_simulated_pristine,
                                                                                            df_simulated_damaged,
                                                                                            df_generated_pristine,
                                                                                            direction=default_direction,
                                                                                            balance_test_set=True,
                                                                                            random_seed=random_seed)

        # do classification - make sure that each run is independent! (pipeline instantiation..)
        clf_results = apply_classifiers(X_train_set, X_test_set, y_train_set, y_test_set,
                                        classifiers=clfs)
        # save classification metrics as df for each classifier
        for name, results in clf_results.items():
            y_true = results["y_true"]["damage_label"]
            y_pred = results["y_pred"]
            y_score = results["y_score"]

            # calculate confusion matrix entries
            conf_matrix = confusion_matrix(y_true, y_pred)
            TN = conf_matrix[0, 0]
            FN = conf_matrix[1, 0]
            FP = conf_matrix[0, 1]
            TP = conf_matrix[1, 1]

            clf_metrics_lst.append({"random_seed": random_seed,
                                    "classifier": name,
                                    "Accuracy": accuracy_score(y_true, y_pred),
                                    "F1-Score": f1_score(y_true, y_pred),
                                    "ROC_AUC": roc_auc_score(y_true, y_score),
                                    "TN": TN,
                                    "FN": FN,
                                    "FP": FP,
                                    "TP": TP})

    # convert to df and calculate mean and std of each column
    df_metrics = pd.DataFrame(clf_metrics_lst)
    stats = []
    for name in df_metrics["classifier"].unique():
        acc_mean = df_metrics.loc[df_metrics["classifier"] == name, "Accuracy"].mean()
        acc_std = df_metrics.loc[df_metrics["classifier"] == name, "Accuracy"].std()

        f1_mean = df_metrics.loc[df_metrics["classifier"] == name, "F1-Score"].mean()
        f1_std = df_metrics.loc[df_metrics["classifier"] == name, "F1-Score"].std()

        auc_mean = df_metrics.loc[df_metrics["classifier"] == name, "ROC_AUC"].mean()
        auc_std = df_metrics.loc[df_metrics["classifier"] == name, "ROC_AUC"].std()

        TN_mean = df_metrics.loc[df_metrics["classifier"] == name, "TN"].mean()
        TN_std = df_metrics.loc[df_metrics["classifier"] == name, "TN"].std()

        FN_mean = df_metrics.loc[df_metrics["classifier"] == name, "FN"].mean()
        FN_std = df_metrics.loc[df_metrics["classifier"] == name, "FN"].std()

        FP_mean = df_metrics.loc[df_metrics["classifier"] == name, "FP"].mean()
        FP_std = df_metrics.loc[df_metrics["classifier"] == name, "FP"].std()

        TP_mean = df_metrics.loc[df_metrics["classifier"] == name, "TP"].mean()
        TP_std = df_metrics.loc[df_metrics["classifier"] == name, "TP"].std()

        stats.append({"Classifier": name,
                      "Accuracy mean": acc_mean, "Accuracy std": acc_std,
                      "F1-Score mean": f1_mean, "F1-Score std": f1_std,
                      "ROC_AUC mean": auc_mean, "ROC_AUC std": auc_std,
                      "TN_mean": TN_mean, "TN_std": TN_std,
                      "FN_mean": FN_mean, "FN_std": FN_std,
                      "FP_mean": FP_mean, "FP_std": FP_std,
                      "TP_mean": TP_mean, "TP_std": TP_std,
                      })
    df_stats = pd.DataFrame(stats)
    return df_stats


def show_clf_result_metrics(clf_results, figure_width_cm=8, figure_height_cm=8, show_confusion_matrices=False):
    for name, results in clf_results.items():
        y_true = results["y_true"]["damage_label"]
        y_pred = results["y_pred"]

        # convert labels: outlier to positive class, inlier to negative class
        #   positive class: damaged -1 -> 1
        #   negative class: pristine 1 -> 0
        y_true_conv = y_true.replace({-1: 1, 1: 0})
        y_pred_conv = pd.Series(y_pred).replace({-1: 1, 1: 0})

        # calculate confusion matrix entries additionally
        conf_matrix = confusion_matrix(y_true_conv, y_pred_conv)
        TN = conf_matrix[0, 0]
        FN = conf_matrix[1, 0]
        FP = conf_matrix[0, 1]
        TP = conf_matrix[1, 1]

        # print classification report
        # analyze results
        print(f"----- {name} ----- ")
        # print accuracy
        print(f"Accuracy: {accuracy_score(y_true_conv, y_pred_conv)}")
        print(f"Balanced accuracy: {balanced_accuracy_score(y_true_conv, y_pred_conv)}")
        print(f"F1 Score: {f1_score(y_true_conv, y_pred_conv)}")
        print(classification_report(y_true_conv, y_pred_conv))
        print(f"confusion matrix components:")
        print(f"True Positive (TP): {TP}")
        print(f"False Positive (FP): {FP}")
        print(f"False Negative (FN): {FN}")
        print(f"True Negative (TN): {TN}")
        print("-----  End of Report ----- \n")

        if show_confusion_matrices:
            fig, ax = init_figure(1, 1,
                                  figure_width_cm=figure_width_cm,
                                  figure_height_cm=figure_height_cm,
                                  use_ratio=False)
            disp = ConfusionMatrixDisplay.from_predictions(y_true_conv, y_pred_conv, ax=ax[0, 0])
            plt.title(name)
            plt.show()


def plot_ROC(clf_results, title=None, figure_width_cm=10, figure_height_cm=8, save_fig=False, save_png=False, fn_extension=""):
    fig, ax = init_figure(1, 1,
                          figure_width_cm=figure_width_cm,
                          figure_height_cm=figure_height_cm,
                          use_ratio=False)
    ax = ax[0, 0]
    for name, results in clf_results.items():
        y_true = results["y_true"]["damage_label"]
        y_score = results["y_score"]

        # convert labels: outlier to positive class, inlier to negative class
        #   positive class: damaged -1 -> 1
        #   negative class: pristine 1 -> 0
        y_true_conv = y_true.replace({-1: 1, 1: 0})

        disp = RocCurveDisplay.from_predictions(y_true_conv, -y_score, pos_label=1, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="green", label="Random guess")
    if title is not None:
        plt.title(title)
    plt.show()

    if save_fig:
        figure_dir = Path("../reports/figures/ROC_curves")
        filename = "ROC_" + fn_extension
        if save_png:
            filename += ".png"
        else:
            filename += ".pdf"
        figure_path = figure_dir / filename
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')

def get_classified_labels(y_test, y_pred, damage_params=False):
    """
    Function to connect damage_labels to classification results.
    Also retrieves label information of damages.
    :param y_test:
    :param y_pred:
    :return:
    """
    # get damage data
    file_path = Path.cwd().parent / "models" / "FEM_model" / "scripts" / "Damage_locations.csv"


    if "experimental" in list(y_test["source"]):
        df_damages = pd.DataFrame([{"name": "DS1", "x": 105, "y": 32.5, "radius": 6.25},
                                    {"name": "DS2", "x": 105, "y": 32.5, "radius": 9.5}])
    else:
        df_damages = pd.read_csv(file_path)
    df_damages = df_damages.rename(columns={"name": "damage_state"})

    y_true_conv = y_test.replace({"damage_label": {-1: 1, 1: 0}})
    y_pred_conv = pd.Series(y_pred).replace({-1: 1, 1: 0})

    df_results = y_true_conv.reset_index().rename(columns={"damage_label": "y_true"})
    df_results.insert(loc=1, column="y_pred", value=y_pred_conv)

    # in case of classification of numerical data insert damage parameters
    if damage_params:
        df_results = pd.merge(df_results, df_damages, how='left', on="damage_state")

    true_pos = df_results.loc[(df_results["y_true"] == 1) & (df_results["y_pred"] == 1)]
    false_neg = df_results.loc[(df_results["y_true"] == 1) & (df_results["y_pred"] == 0)]
    false_pos = df_results.loc[(df_results["y_true"] == 0) & (df_results["y_pred"] == 1)]
    true_neg = df_results.loc[(df_results["y_true"] == 0) & (df_results["y_pred"] == 0)]

    # get location and sizes of damages
    # apply function -> to each entry add damage parameters

    return {"TP": true_pos, "FN": false_neg, "FP": false_pos, "TN": true_neg}


def plot_classified_damages(classified_labels, save_figure=False, figure_width_cm=12, figure_height_cm=None, fn_extension=""):
    """
    Plots classified damages.
    Colorcode shows if correctly classified (TN - True Negative) or incorrectly classified (FP - False Positive, missed)
    :param clf_result_labels: dict with classified labels of SINGLE classifier
    :param target_path:
    :param figure_width_cm:
    :return:
    """
    fig, ax = init_figure(1, 1,
                          figure_width_cm=figure_width_cm,
                          figure_height_cm=figure_height_cm,
                          use_ratio=False)
    ax = ax[0, 0]
    ax.set_aspect("equal")

    # get damage data
    file_path = Path.cwd().parent / "models" / "FEM_model" / "scripts" / "Damage_locations.csv"
    df_damages = pd.read_csv(file_path)

    # global sensor coordinates
    sensor_size = 6
    sensor_locations_global = np.array([[195, 352.5], [250, 352.5], [305, 352.5],
                                        [195, 297.5], [250, 297.5], [305, 297.5],
                                        [195, 242.5], [250, 242.5], [305, 242.5]])
    df_TP = classified_labels["TP"]
    df_FN = classified_labels["FN"]

    TP_damage_states = df_TP["damage_state"].unique()
    FN_damage_states = df_FN["damage_state"].unique()
    # plot True Positive classifications in green (hit)
    for row in range(df_TP.index.size):
        x_glob = df_TP['x'].iloc[row] + 140
        y_glob = df_TP['y'].iloc[row] + 187.5
        radius = df_TP['radius'].iloc[row]
        TP_damage_state = df_TP['damage_state'].iloc[row]
        if TP_damage_state not in FN_damage_states:
            ax.add_patch(patches.Circle((x_glob, y_glob), radius,
                                        facecolor='green',
                                        edgecolor='darkgreen',
                                        alpha=0.4))
    # manually create legend
    green_patch = mpatches.Patch(facecolor='green',
                                 edgecolor='darkgreen',
                                 alpha=0.4,
                                 label="True Positive (hit)")

    # plot False Negative classifications in red (miss)
    for row in range(df_FN.index.size):
        x_glob = df_FN['x'].iloc[row] + 140
        y_glob = df_FN['y'].iloc[row] + 187.5
        radius = df_FN['radius'].iloc[row]
        FN_damage_state = df_FN['damage_state'].iloc[row]
        if FN_damage_state in TP_damage_states:
            ax.add_patch(patches.Circle((x_glob, y_glob), radius,
                                        facecolor='orange',
                                        edgecolor='darkorange',
                                        alpha=0.4))
        else:
            ax.add_patch(patches.Circle((x_glob, y_glob), radius,
                                        facecolor='red',
                                        edgecolor='darkred',
                                        alpha=0.4))
    # manually create legend
    red_patch = mpatches.Patch(facecolor='red',
                                 edgecolor='darkred',
                                 alpha=0.4,
                                 label="False Negative (miss)")
    orange_patch = mpatches.Patch(facecolor='orange',
                               edgecolor='darkorange',
                               alpha=0.4,
                               label="Both")

    # add sensor locations
    for sensor_xy in sensor_locations_global:
        ax.add_patch(patches.Rectangle((sensor_xy[0] - sensor_size / 2, sensor_xy[1] - sensor_size / 2),
                                       sensor_size,
                                       sensor_size,
                                       facecolor='black',
                                       alpha=0.6,
                                       ec='black'))
    # handle and position legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([green_patch, red_patch, orange_patch])
    fig.legend(handles=handles, bbox_to_anchor=(1.13, 0.90), loc='upper right')

    ax.set_xlim([140, 360])
    ax.set_ylim([187.5, 380])

    ax.set_xlabel(r'${}_{I}x$ in mm', labelpad=10.0)
    ax.set_ylabel(r'${}_{I}y$ in mm', labelpad=10.0)

    plt.show()
    if save_figure:
        target_dir = Path("../reports/figures/classified_damages")
        filename = f"Classifed_damages_w{figure_width_cm}cm"
        if fn_extension is not None:
            filename += f"_{fn_extension}"
        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')

def stacked_hist_misclass_radius(classified_labels, bins=20, save_figure=False, figure_width_cm=12, figure_height_cm=None, fn_extension=""):
    fig, ax = init_figure(1, 1,
                          figure_width_cm=figure_width_cm,
                          figure_height_cm=figure_height_cm,
                          use_ratio=False)
    ax = ax[0, 0]
    # prepare dataset -> merge dfs with new column "classification" holding TP and FN
    df_TP = classified_labels["TP"]
    df_FN = classified_labels["FN"]
    df_TP["classification"] = "TP"
    df_FN["classification"] = "FN"
    df = pd.concat([df_TP, df_FN], ignore_index=True)

    # plot hits (TP) and misses (FN) as stacked histogram over damage radius
    ax_sns = sns.histplot(df,
                          x="radius",
                          hue="classification",
                          multiple="stack",
                          stat="count",
                          bins=bins)
    ax_sns.set_xlabel("radius in mm")
    ax_sns.minorticks_on()
    ax_sns.tick_params(axis='y', which='minor', left=False)

    plt.show()
    if save_figure:
        target_dir = Path("../reports/figures/classified_damages")
        filename = f"Classifed_damages_w{figure_width_cm}cm"
        if fn_extension is not None:
            filename += f"_{fn_extension}"
        figure_path = target_dir / filename
        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')

def plot_score_over_n_olcs(df, y_data="Balanced accuracy", figure_width_cm=10, figure_height_cm=None, save_figure=False, fn_extension=""):
    fig, ax = init_figure(1, 1,
                          figure_width_cm=figure_width_cm,
                          figure_height_cm=figure_height_cm,
                          use_ratio=False)
    ax = ax[0, 0]
    if "u" in df.columns:
        for u in df["u"].unique():
            df_sub = df[df["u"] == u]
            df_sub.plot(ax=ax, x="n_olcs", y=y_data, marker="o", markersize=3,
                        linestyle="--", label=f"u = {u}")
        ax.set_ylabel(y_data)
    else:
        df.plot(ax=ax, x="n_olcs", y=y_data, marker="o", markersize=3, linestyle="--")
        ax.set_ylabel("Classification score")
    ax.set_xlabel("Number of objective loadcases")
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', left=False)
    ax.grid(which="major", linewidth=0.5)

    plt.show()
    if save_figure:
        target_dir = Path("../reports/figures/Balanced_acc_over_OLCs")
        filename = f"Balanced_acc_w{figure_width_cm}cm"
        if fn_extension is not None:
            filename += f"_{fn_extension}"
        figure_path = target_dir / filename
        figure_path = target_dir / filename
        for suffix in [".pdf", ".png"]:
            fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches='tight')


def get_principal_strains(df_data):
    """
    transform measured strain values of each sensor to principal strains and
    angle FROM grid direction TO principal strain axis.

    :param df: n x (features + lables)
    :return: Dataframe n x (principal components + lables)
    """
    df = df_data.copy()
    labels = ["loadcase", "damage_label", "damage_state", "source", "x", "y", "r"]
    sensor_labels, direction_map, notation_map, grid_position = get_sensor_information()
    principal_values = []
    for sensor_nr, sensor_label in sensor_labels.items():
        # get strains a, b, c from df
        a = df[sensor_label["desc_x"]]
        b = df[sensor_label["desc_xi"]]
        c = df[sensor_label["desc_y"]]
        # elementwise calculate principal strains and angle theta
        strain_P = (a + c)/2 + 1/np.sqrt(2) * np.sqrt((a - b)**2 + (b - c)**2)
        strain_Q = (a + c) / 2 - 1 / np.sqrt(2) * np.sqrt((a - b) ** 2 + (b - c) ** 2)
        # handle angle ambiguity
        Z = 2*b - a - c
        N = a - c
        # handle angle ambiguity
        phi = 1/2 * np.arctan2(Z, N)  #range [-pi, pi], phi is angle from grid a to principal axis
        principal_values.append(pd.DataFrame({f"P_{sensor_nr}": strain_P, f"Q_{sensor_nr}": strain_Q, f"phi_{sensor_nr}": phi}))
    df_principal = pd.concat(principal_values, axis=1)
    return pd.concat([df_principal, df[labels]], axis=1)


def correct_sensor_misalignment(df, corrections: list):
    """
    rotate given strain measurement values of all 3 directions of a given sensor with
    the correction angle.
    :param df:
    :param sensor_nr:
    :param corrections: list of dicts (sensor_nr, correction_angle in degrees)
    :return: df with corrected rotational misalignment
    """
    df_corrected = df.copy()
    sensor_labels, direction_map, notation_map, grid_position = get_sensor_information()
    # calculate principle strains and angle
    df_principal = get_principal_strains(df_data=df)
    # perform corrections
    for correction in corrections:
        sensor_nr = correction["sensor_nr"]
        correction_angle = correction["correction_angle"]

        strain_P_name = f"P_{sensor_nr}"
        strain_Q_name = f"Q_{sensor_nr}"
        phi_name = f"phi_{sensor_nr}"

        strain_P = df_principal[strain_P_name]
        strain_Q = df_principal[strain_Q_name]
        phi = df_principal[phi_name]

        # get new angle from grid to principle strain with correction angle
        new_phi_a = phi + np.radians(correction_angle)
        new_phi_b = phi + np.radians(correction_angle) - np.pi/4
        new_phi_c = phi + np.radians(correction_angle) - np.pi/2

        # calculate strain for grid a, b(+45) and c(+90)
        feature_a = sensor_labels[sensor_nr]["desc_x"]
        feature_b = sensor_labels[sensor_nr]["desc_xi"]
        feature_c = sensor_labels[sensor_nr]["desc_y"]

        df_corrected[feature_a] = (strain_P + strain_Q) / 2 + (strain_P - strain_Q) / 2 * np.cos(-2 * new_phi_a)
        df_corrected[feature_b] = (strain_P + strain_Q) / 2 + (strain_P - strain_Q) / 2 * np.cos(-2 * new_phi_b)
        df_corrected[feature_c] = (strain_P + strain_Q) / 2 + (strain_P - strain_Q) / 2 * np.cos(-2 * new_phi_c)

    return df_corrected

if __name__ == '__main__':
    pass