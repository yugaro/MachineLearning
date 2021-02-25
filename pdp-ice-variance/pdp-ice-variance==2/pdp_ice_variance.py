import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import partial_dependence
from sklearn.datasets import load_breast_cancer
np.random.seed(0)
plt.rcParams['font.size'] = 18


def data_cleansing(dataset):
    # # data cleansing
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y_train = dataset.target
    feature_name = [X.columns[7], X.columns[20], X.columns[21],
                    X.columns[24], X.columns[27], X.columns[28]]
    X_train = X[feature_name]
    return X_train, y_train, feature_name


def derivative_func(y_prior, y_post, h):
    # # compute derivative
    return (y_post - y_prior) / h


def plot_analysis(grid_value, predicted_dvar, feature_name):
    # # plot the variance of derivative of ice for each feature

    # preparing
    grid_value = grid_value[: -1]

    # plot data
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for k in range(len(feature_name)):
        ax = fig.add_subplot(
            2, len(feature_name) / 2 + len(feature_name) % 2, k + 1)
        ax.plot(grid_value, predicted_dvar[k])
        ax.set_xlabel(feature_name[k], fontsize=18)
        ax.set_ylabel('variance of deriv. ice', fontsize=18)
    fig.suptitle("feature - variance of deriv. ice (benign prob.)")
    fig.savefig('./images/test01.pdf')


def plot_analysis2(grid_value, predicted_dvar, predicted_dmean, feature_name):
    # # plot derivative pdp with the variance of derivative of ice

    # preparing
    grid_value = grid_value[: -1]
    predicted_std = np.sqrt(predicted_dvar)

    # plot data
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for k in range(len(feature_name)):
        ax = fig.add_subplot(
            2, len(feature_name) / 2 + len(feature_name) % 2, k + 1)
        ax.plot(grid_value, predicted_dmean[k], color='blue', label='mean')
        ax.fill_between(grid_value, predicted_dmean[k] + 2 * predicted_std[k],
                        predicted_dmean[k] - 2 * predicted_std[k], alpha=.2, color='blue', label='std')
        ax.set_xlabel(feature_name[k], fontsize=18)
        ax.set_ylabel('deriv. pdp (benign prob.)', fontsize=18)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
    fig.suptitle("feature - deriv. pdp (benign prob.)")
    fig.savefig('./images/test02.pdf')


def plot_analysis3(grid_value, pdp_data, predicted_dmean, predicted_dvar, feature_name):
    # # plot the prob - pdp with variance

    # preparing
    h = grid_value[1] - grid_value[0]
    predicted_std = np.sqrt(predicted_dvar)

    # plot data
    pdp_upper = np.zeros((pdp_data.shape[0], pdp_data.shape[1]))
    pdp_bottom = np.zeros((pdp_data.shape[0], pdp_data.shape[1]))
    pdp_upper[:, 0] = pdp_data[:, 0]
    pdp_bottom[:, 0] = pdp_data[:, 0]

    for i in range(pdp_data.shape[1] - 1):
        pdp_upper[:, i + 1] = pdp_data[:, i] + h * \
            (predicted_dmean[:, i] + 2 * predicted_std[:, i])
        pdp_bottom[:, i + 1] = pdp_data[:, i] + h * \
            (predicted_dmean[:, i] - 2 * predicted_std[:, i])

    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for k in range(len(feature_name)):
        ax = fig.add_subplot(2, len(feature_name) / 2 +
                             len(feature_name) % 2, k + 1)
        ax.plot(grid_value, pdp_data[k], color='blue', label='mean')
        ax.fill_between(
            grid_value, pdp_upper[k], pdp_bottom[k], alpha=.2, color='blue', label='std')
        ax.set_xlabel(feature_name[k], fontsize=18)
        ax.set_ylabel('benign prob.', fontsize=18)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
    fig.suptitle("feature - benign prob.")
    fig.savefig('./images/test03.pdf')


class visible_ice_variance:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        # # model fitting
        self.model.fit(X_train, y_train)

    def analyze_ice_derivative_variance(self, X_train, grid_resolution):
        # # analyze variances of derivative of ice

        predicted_dvar = np.zeros((X_train.shape[1], grid_resolution - 1))
        predicted_dmean = np.zeros((X_train.shape[1], grid_resolution - 1))
        pdp_data = np.zeros((X_train.shape[1], grid_resolution))
        for k in range(X_train.shape[1]):
            # get the trajectories of ice for each features
            result_ice = partial_dependence(
                self.model, X_train, features=[k], percentiles=(0, 1), grid_resolution=grid_resolution, kind='both')

            # get values of grid and data of ice
            grid_value = np.array(result_ice['values'][0])
            predicted_point = result_ice['individual'][0]
            predicted_point_mean = result_ice['average'][0]
            pdp_data[k] = predicted_point_mean

            predicted_derivative = np.zeros(
                (predicted_point.shape[0], predicted_point.shape[1] - 1))
            predicted_derivative_mean = np.zeros(
                predicted_point_mean.shape[0] - 1)

            # get gaps between grid
            h = grid_value[1] - grid_value[0]
            for i in range(predicted_point.shape[0]):
                for j in range(predicted_point.shape[1] - 1):
                    # get values of prior and postrior ice
                    y_prior = predicted_point[i][j]
                    y_post = predicted_point[i][j + 1]

                    # get mean values of prior and postrior ice
                    y_prior_mean = predicted_point_mean[j]
                    y_post_mean = predicted_point_mean[j + 1]

                    # compute the derivative of ice based on derivative func
                    predicted_derivative[i][j] = derivative_func(
                        y_prior, y_post, h)

                    # compute the mean derivative of ice based on derivative func
                    predicted_derivative_mean[j] = derivative_func(
                        y_prior_mean, y_post_mean, h)

            # get variances of derivatice of ice for each features
            predicted_dvar[k] = np.var(predicted_derivative, axis=0)

            # get mean of derivatice of ice for each features
            predicted_dmean[k] = predicted_derivative_mean

        return grid_value, predicted_dvar, predicted_dmean, pdp_data


if __name__ == '__main__':
    # get dataset and implement cleansing
    dataset = load_breast_cancer()
    X_train, y_train, feature_name = data_cleansing(dataset)

    # fit model
    rfc = RandomForestClassifier()
    lr = LogisticRegression()
    viv = visible_ice_variance(model=rfc)
    viv.fit(X_train, y_train)

    # set grid resulution and get grid values and prediction variance
    grid_resolution = 400
    grid_value, predicted_dvar, predicted_dmean, pdp_data = viv.analyze_ice_derivative_variance(
        X_train=X_train, grid_resolution=grid_resolution)

    # plot data
    plot_analysis(grid_value, predicted_dvar, feature_name)

    # plot data2
    plot_analysis2(grid_value, predicted_dvar, predicted_dmean, feature_name)

    # plot data3
    plot_analysis3(grid_value, pdp_data,
                   predicted_dmean, predicted_dvar, feature_name)
