import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.datasets import load_breast_cancer
np.random.seed(0)
plt.rcParams['font.size'] = 18


def data_cleansing(dataset):
    # data cleansing
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y_train = dataset.target
    feature_name = [X.columns[7], X.columns[20], X.columns[21],
                    X.columns[24], X.columns[27], X.columns[28]]
    X_train = X[feature_name]
    return X_train, y_train, feature_name


def derivative_func(y_prior, y_post, h):
    # # compute derivative
    return (y_post - y_prior) / h


def plot_analysis(grid_value, predicted_var, feature_name):
    # 特徴量ごとにiceの微分の分散をプロットplot the variance of derivative of ice for each feature
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for k in range(len(feature_name)):
        ax = fig.add_subplot(
            2, len(feature_name) / 2 + len(feature_name) % 2, k + 1)
        ax.plot(grid_value, predicted_var[k])
        ax.set_xlabel(feature_name[k], fontsize=20)
        ax.set_ylabel('variance')
    fig.savefig('./variance_ice.pdf')


class visible_ice_variance:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        # # model fitting
        self.model.fit(X_train, y_train)

    def analyze_ice_derivative_variance(self, X_train, grid_resolution):
        # # analyze variances of derivative of ice

        predicted_var = np.zeros((X_train.shape[1], grid_resolution))
        for k in range(X_train.shape[1]):
            # get the trajectories of ice for each features
            result_ice = partial_dependence(
                self.model, X_train, features=[k], percentiles=(0, 1), grid_resolution=grid_resolution, kind='individual')

            # get values of grid and data of ice
            grid_value = np.array(result_ice['values'][0])
            predicted_point = result_ice['individual'][0]

            predicted_derivative = np.zeros(
                (predicted_point.shape[0], predicted_point.shape[1]))

            # get gaps between grid
            h = grid_value[1] - grid_value[0]
            for i in range(predicted_point.shape[0]):
                for j in range(predicted_point.shape[1] - 1):
                    # get values of prior and postrior ice
                    y_prior = predicted_point[i][j]
                    y_post = predicted_point[i][j + 1]

                    # compute the derivative of ice based on derivative func
                    predicted_derivative[i][j] = derivative_func(
                        y_prior, y_post, h)

            # get variances of derivatice of ice for each features
            predicted_var[k] = np.var(predicted_derivative, axis=0)

        return grid_value, predicted_var


if __name__ == '__main__':
    # get dataset and implement cleansing
    dataset = load_breast_cancer()
    X_train, y_train, feature_name = data_cleansing(dataset)

    # fit model
    RFC = visible_ice_variance(model=RandomForestClassifier())
    RFC.fit(X_train, y_train)

    # set grid resulution and get grid values and prediction variance
    grid_resolution = 200
    grid_value, predicted_var = RFC.analyze_ice_derivative_variance(
        X_train=X_train, grid_resolution=grid_resolution)

    # plot data
    plot_analysis(grid_value, predicted_var, feature_name)
