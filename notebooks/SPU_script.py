"""Class definitions for damage identification
SPU class and damage identification class
Author: student k1256205@students.jku.at
Created: 05/06/2023
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import MDS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump, load

from support import *
from support_damage_detection import *

def load_data(df, direction=["desc_x", "desc_y", "desc_xi"]):
    _, direction_map, notation_map, _ = get_sensor_information()
    features = []
    for direct in direction:
        features.extend(direction_map[direct])
    #df = df.set_index("loadcase")
    df_features = df[features]
    df_labels = df.drop(features, axis=1)
    df_labels["damage_label"] = df_labels["damage_label"].replace({0: 1, 1: -1})
    return df_features, df_labels

def preprocess(df_features, pipeline, transform=False):
    features = df_features.columns
    if transform:
        df_features[features] = pipeline.transform(df_features)
    else:
        df_features[features] = pipeline.fit_transform(df_features)
    return df_features

def isolate_features(df_features, df_labels):
    data_sets = []
    for feature in df_features.columns:
        df_X = df_features.copy()
        y = df_X.pop(feature)
        data_sets.append({"feature": feature, "df_X": df_X, "df_y": pd.concat([y, df_labels], axis=1)})
    return data_sets, df_labels

def regressor_training_procedure(df_X, df_y, feature, reg_model, params,
                                 param_grid=None, init_params=None, perform_CV=False):
    # this is called seperatley for each feature
    # perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=0)

    if perform_CV:
        reg = reg_model(**init_params)
        scoring = make_scorer(accuracy_score)
        search = GridSearchCV(reg, param_grid) #R2 score is used as default for regression
        search.fit(X_train, y_train[feature])
        print(f"Cross-Validation of Reg. {feature}")
        print(f"CV score: {search.best_score_}")
        print(f"Best params: {search.best_params_}")
        init_params.update(search.best_params_)
        params = init_params
    reg = reg_model(**params)

    # train regressor with chosen hyperparameters
    reg.fit(X_train, y_train[feature])

    # get error on pristine data
    y_pred = reg.predict(X_test)

    # apply error metric
    delta_pristine = error_metric(y_test[feature], y_pred)

    return feature, delta_pristine, y_test, reg
    #return reg, params, delta_pristine, feature

def regressor_transformation_procedure(df_X, y_test, regressor):
    # gets the fitted regressor from training procedure
    y_pred = regressor.predict(df_X)
    return error_metric(y_test, y_pred)

def error_metric(y_test, y_pred):
    # common error metric to use in training and transformation procedure
    return np.sqrt((y_test - y_pred)**2)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
from pathlib import Path
import json

class SPU:
    # Strain Prediciton Unit
    def __init__(self, feature, regressor, params=None, random_state=0):
        self.feature = feature
        self.regressor = regressor
        self.filename = f"{regressor.__name__}.json"
        self.filename_cv = f"{regressor.__name__}_CV.json"
        self.trained_regressor = None
        self.random_state = random_state
        if params is None:
            try:
               self.params = self._read_params(self.filename)[feature]  #should only be one dict of params
            except:
                self.params = params
        else:
            self.params = params
            self._write_params(params, self.filename)

    def split_data(self, df_X, df_target, test_size=0.2, only_label_separation=False):
        if test_size == 0.0 or only_label_separation:
            # do not perform split - only label separation (e.g. for test data)
            X_test = df_X.copy()
            y_labels_test = df_target.copy()
            y_test = y_labels_test.pop(self.feature)
            X_train = None
            y_train = None
            y_labels_train = None
        else:
            X_train, X_test, y_labels_train, y_labels_test = train_test_split(df_X, df_target,
                                                                              test_size=test_size,
                                                                              random_state=self.random_state)
            y_train = y_labels_train.pop(self.feature)
            y_test = y_labels_test.pop(self.feature)
        return X_train, X_test, y_train, y_test, y_labels_train, y_labels_test  # expected: lables only contain lables

    def train(self, X_train, y_train):
        # read params from csv file
        # if it doesnt exist or has no entries -> use default parameters
        assert self.params is not None, "No parameters have been specified."
        reg = self.regressor(**self.params)
        reg.fit(X_train, y_train)
        self.trained_regressor = reg

    def transform(self, X_test, y_test):
        assert self.trained_regressor is not None, "Regressor has not been fitted yet."
        y_pred = self.trained_regressor.predict(X_test)
        score_r2 = r2_score(y_test, y_pred)  # dataset dependent! only makes sense for training data and novelty model evaluation
        score_mse = mean_squared_error(y_test, y_pred)
        score_mape = mean_absolute_percentage_error(y_test, y_pred)
        # error metric for prediction deltas
        delta = np.sqrt((y_test - y_pred)**2)
        return delta, score_r2, score_mse, score_mape

    def hyperparameter_tuning(self, X_train, y_train, init_params, param_grid):
        reg = self.regressor(**init_params)
        search = GridSearchCV(reg, param_grid)
        search.fit(X_train, y_train)
        # save
        init_params.update(search.best_params_)
        params = init_params
        self._write_params(params, self.filename_cv) #TODO: this should be written to a seperate file, to allow parallel work
        return params, search.best_score_

    def _write_params(self, params, filename):
        if Path(filename).is_file():
            params_dict = self._read_params(filename)  # complete set of parameters -> all features
            # update the parameters of the specific feature
            params_dict.update({self.feature: params})  # dict of a dict
        else:
            params_dict = {self.feature: params}
        with open(filename, "w") as file:
            json.dump(params_dict, file)

    def _read_params(self, filename):
        with open(filename, "r") as file:
            params_dict = json.load(file)
        return params_dict


class SensorGridDamageIdentification:
    # Damage Identification Model
    #TODO include method that handles preprocessing for both training and testing data
    def __init__(self, direction, spu_class, preprocess_pipeline=None, random_state=0):
        self.direction = direction
        self.spu_class = spu_class
        self.reg_model = None
        self.trained_SPUs = None
        self.preprocess_pipeline = preprocess_pipeline
        self.direction_map = {'desc_y': ['V_1_1', 'V_2_1', 'V_3_1', 'V_1_2', 'V_2_2', 'V_3_2', 'V_1_3', 'V_2_3', 'V_3_3'],
                              'desc_x': ['H_1_1', 'H_1_2', 'H_1_3', 'H_2_1', 'H_2_2', 'H_2_3', 'H_3_1', 'H_3_2', 'H_3_3'],
                              'desc_xi': ['VH_1_1', 'VH_1_2', 'VH_1_3', 'VH_2_1', 'VH_2_2', 'VH_2_3', 'VH_3_1', 'VH_3_2', 'VH_3_3']}
        self.features = []
        self.random_state = random_state
        for each in direction:
            self.features.extend(self.direction_map[each])

        # prediction attributes
        self.benchmark = None
        self.threshold = None

    def hyperparameter_tuning(self, train_data, reg_model, init_params, param_grid):
        # find hyperparameters for each SPU individually based on train data
        # save parameters, so that successive execution of fit method can use them (in fit: params=None)
        # do preprocessing on data if a pipeline is passed
        if self.preprocess_pipeline is not None:
            train_data[self.features] = self.preprocess_pipeline.fit_transform(train_data[self.features])
        self.reg_model = reg_model.__name__
        df_features, df_labels = self._load_data(train_data)
        data_pairs = self._isolate_features(df_features, df_labels)
        for feature, data_pair in tqdm(data_pairs.items(), "Hyperparameter tuning: "):
            df_X = data_pair["df_X"]
            df_labels = data_pair["df_labels"]

            # create SPU instance for the specific feature
            spu = self.spu_class(feature=feature, regressor=reg_model, params=None, random_state=self.random_state)
            # split data
            X_train, _, y_train, _, y_labels_train, _ = spu.split_data(df_X, df_labels)
            # perform hyperparameter search -> method handles writing params to param file
            params, CV_best_score = spu.hyperparameter_tuning(X_train, y_train, init_params, param_grid)


    def fit(self, train_data, reg_model, params=None):
        # If params are not specified or params=None, instantiation of spu tries to read the params file,
        # containing the latest params
        self.reg_model = reg_model.__name__
        # do preprocessing on data if a pipeline is passed
        if self.preprocess_pipeline is not None:
            train_data[self.features] = self.preprocess_pipeline.fit_transform(train_data[self.features])

        #load params option -> covered by SPU.__init__() method
        df_features, df_labels = self._load_data(train_data)
        data_pairs = self._isolate_features(df_features, df_labels)
        train_results = {}
        r2_scores = {}
        mse_scores = {}
        mape_scores = {}
        regressors = {}
        for feature, data_pair in tqdm(data_pairs.items(), "Regressor fitting: "):
            df_X = data_pair["df_X"]
            df_labels = data_pair["df_labels"]

            # create SPU instance for the specific feature
            spu = self.spu_class(feature=feature, regressor=reg_model, params=params, random_state=self.random_state)
            # split data
            X_train, X_test, y_train, y_test, y_labels_train, y_labels_test = spu.split_data(df_X, df_labels)
            # train classifier
            spu.train(X_train, y_train)
            # get delta on training data
            delta_training, score_r2_train, score_mse_train, score_mape = spu.transform(X_test, y_test)
            # store data
            train_results[feature] = delta_training
            r2_scores[feature] = score_r2_train
            mse_scores[feature] = score_mse_train
            mape_scores[feature] = score_mape
            regressors[feature] = spu

        self.trained_SPUs = regressors
        # pickle dump trained regressors
        dump(regressors, f"{self.reg_model}.joblib")
        df_train_results = pd.DataFrame.from_dict(train_results)
        df_train_results = pd.concat([df_train_results, y_labels_test[["loadcase", "damage_state", "damage_label", "source"]]], axis=1)
        return df_train_results, r2_scores, mse_scores, mape_scores

    def transform(self, test_data, trained_model=None):
        if trained_model is not None:
            # it should be sufficient to only call transform when this flag is set
            self.trained_SPUs = load(f"{trained_model}.joblib") #DEBUG: should load a dict of trained regressors
        else:
            # apply preprocessing on data if a pipeline is passed -> only available when pipeline fas fitted before!
            if self.preprocess_pipeline is not None:
                test_data[self.features] = self.preprocess_pipeline.transform(test_data[self.features])
        df_features, df_labels = self._load_data(test_data)
        data_pairs = self._isolate_features(df_features, df_labels)
        transform_results = {}
        for feature, data_pair in data_pairs.items():
            df_X = data_pair["df_X"]
            df_labels = data_pair["df_labels"]
            spu = self.trained_SPUs[feature]
            _, X_trans, _, y_trans, _, y_labels_trans = spu.split_data(df_X, df_labels, only_label_separation=True)
            delta_transform, _, _, _ = spu.transform(X_trans, y_trans)
            transform_results[feature] = delta_transform

        df_transform_results = pd.DataFrame.from_dict(transform_results)
        df_transform_results = pd.concat([df_transform_results, y_labels_trans[["loadcase", "damage_state", "damage_label", "source"]]], axis=1)
        return df_transform_results

    def predict(self, df_train, df_test, margin=0.5):
        # this method only handles transformed data!
        # get pristine samples to determine threshold
        sample_pristine = df_test[df_test["damage_state"] == "pristine"].sample(n=1, random_state=self.random_state)
        X_sample_pristine = sample_pristine[self.features]

        # remove sample from test set
        df_test = df_test.drop(sample_pristine.index).reset_index(drop=True)
        X_test = df_test[self.features]
        y_test = df_test[["loadcase", "damage_state", "damage_label", "source"]]

        # prepare train set
        X_train = df_train[self.features]

        # determine threshold
        self.benchmark = X_train.median(axis=1).mean()  # use median, because robust to prediction outliers
        mean_sample = X_sample_pristine.mean(axis=1)  # use mean, because it is more sensitive to outliers -> actual objective
        self.threshold = max(self.benchmark, mean_sample.mean()) * (1 + margin) #regularization based on standard deviation?

        # calculate damage_index and predict label
        mean_test = X_test.mean(axis=1)
        y_pred = np.where(mean_test > self.threshold, -1, 1)

        return y_pred, y_test, mean_test


    def _load_data(self, df):
        df_features = df[self.features]
        df_labels = df.drop(self.features, axis=1)
        df_labels["damage_label"] = df_labels["damage_label"].replace({0: 1, 1: -1})
        return df_features, df_labels

    def _isolate_features(self, df_features, df_labels):
        data_pairs = {}
        for feature in df_features.columns:
            df_X = df_features.copy()
            y = df_X.pop(feature)
            data_pairs[feature] = {"df_X": df_X, "df_labels": pd.concat([y, df_labels], axis=1)}
        return data_pairs


if __name__ == '__main__':
    pass