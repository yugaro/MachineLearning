import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
np.random.seed(0)
plt.rcParams['font.size'] = 18

def data_cleansing(dataset):
    # select features
    X_dataset = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    feature_names = [X_dataset.columns[7], X_dataset.columns[20], X_dataset.columns[21],
                     X_dataset.columns[24], X_dataset.columns[27], X_dataset.columns[28]]

    # normalize dataset (範囲: [0, 1])
    mms = MinMaxScaler(feature_range=(0, 1), copy=True)
    X_train = mms.fit_transform(X_dataset[feature_names])

    # extract label
    y_train = dataset.target

    return X_train, y_train, feature_names


def derivative_func(prior_data, post_data, h):
    # compute derivative
    return (post_data - prior_data) / h


def analyze_var_dice(model, X_train, grid_resolution):
    # analyze variances of derivative of ice
    grid_value = np.zeros((X_train.shape[1], grid_resolution))
    pdp_data = np.zeros((X_train.shape[1], grid_resolution))
    ice_data = np.zeros(
        (X_train.shape[1], X_train.shape[0], grid_resolution))
    dpdp_data = np.zeros((X_train.shape[1], grid_resolution - 1))
    dice_data = np.zeros(
        (X_train.shape[1], X_train.shape[0], grid_resolution - 1))
    var_dice_data = np.zeros((X_train.shape[1], grid_resolution - 1))

    for k in range(X_train.shape[1]):
        # get the trajectories of ice for each features
        pdp_ice_data = partial_dependence(
            estimator=model, X=X_train, features=[k], percentiles=(0, 1), grid_resolution=grid_resolution, method='brute', kind='both')

        # get values of grid and data of ice
        grid_value[k] = np.array(pdp_ice_data['values'][0])
        pdp_data[k] = pdp_ice_data['average'][0]
        ice_data[k] = pdp_ice_data['individual'][0]

        # get gaps between grid
        h = grid_value[k][1] - grid_value[k][0]
        for i in range(grid_resolution - 1):
            # compute the derivative of ice based on derivative func
            dpdp_data[k][i] = derivative_func(
                pdp_data[k][i], pdp_data[k][i + 1], h)
            dice_data[k][:, i] = derivative_func(
                ice_data[k][:, i], ice_data[k][:, i + 1], h)
        var_dice_data[k] = np.var(dice_data[k], axis=0)

    return grid_value, pdp_data, ice_data, dpdp_data, dice_data, var_dice_data

def plot_var_dice(grid_value, var_dice_data, feature_names):

    # plot data
    fig = plt.figure(figsize=(15, 9))
    for k in range(len(feature_names)):
        ax = fig.add_subplot(2, len(feature_names) / 2 +
                             len(feature_names) % 2, k + 1)
        ax.plot(grid_value[k][: -1], var_dice_data[k],
                color='c', label='Var. d-ICE plot')
        ax.set_xlabel(feature_names[k])
        ax.set_ylabel('var. d-ICE')
        ax.set_ylim(-20, var_dice_data.max() + 20)
        ax.grid(linestyle='dotted', lw=0.5)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Feature - Variance of d-ICE")
    fig.savefig('./images/var_dice.png')


def pdp_upper_bottom(grid_value, pdp_data, dpdp_data, var_dice_data):
    pdp_data_upper = np.zeros((pdp_data.shape[0], pdp_data.shape[1]))
    pdp_data_bottom = np.zeros((pdp_data.shape[0], pdp_data.shape[1]))
    pdp_data_upper[:, 0] = pdp_data[:, 0]
    pdp_data_bottom[:, 0] = pdp_data[:, 0]

    h = grid_value[:, 1] - grid_value[:, 0]
    std_dice_data = np.sqrt(var_dice_data)

    for i in range(pdp_data.shape[1] - 1):
        pdp_data_upper[:, i + 1] = pdp_data[:, i] + h * \
            (dpdp_data[:, i] + 2 * std_dice_data[:, i])
        pdp_data_bottom[:, i + 1] = pdp_data[:, i] + h * \
            (dpdp_data[:, i] - 2 * std_dice_data[:, i])

    return pdp_data_upper, pdp_data_bottom


def plot_pdp_std(grid_value, pdp_data, pdp_data_upper, pdp_data_bottom, feature_names):
    fig = plt.figure(figsize=(15, 9))
    for k in range(len(feature_names)):
        ax = fig.add_subplot(2, len(feature_names) / 2 +
                             len(feature_names) % 2, k + 1)

        ax.plot(grid_value[k], pdp_data[k], color='red', label='PDP')
        ax.fill_between(grid_value[k], pdp_data_upper[k],
                        pdp_data_bottom[k], color='c', alpha=.3, label='STD')
        ax.set_xlabel(feature_names[k])
        ax.set_ylabel('benign prob.')
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
        ax.grid(linestyle='dotted', lw=0.5)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Feature - Benign Prob. (with STD.)")
    fig.savefig('./images/pdp_std.png')

# read dataset
dataset = load_breast_cancer()

# extract training dataset
X_train, y_train, feature_names = data_cleansing(dataset)

# lean model by random forest
rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)

# learn model by logistic regression
lg = LogisticRegression()
lg.fit(X_train, y_train)

# set grid resulution and get grid values and prediction variance
grid_resolution = 200
# grid_value, pdp_data, ice_data, dpdp_data, dice_data, var_dice_data = analyze_var_dice(
#     model=rfc, X_train=X_train, grid_resolution=grid_resolution)
grid_value, pdp_data, ice_data, dpdp_data, dice_data, var_dice_data = analyze_var_dice(
    model=lg, X_train=X_train, grid_resolution=grid_resolution)

# plot data
plot_var_dice(grid_value, var_dice_data, feature_names)

# compute upper and bottom of pdp
pdp_data_upper, pdp_data_bottom = pdp_upper_bottom(
    grid_value, pdp_data, dpdp_data, var_dice_data)

# visualize pdp with std
plot_pdp_std(grid_value, pdp_data, pdp_data_upper, pdp_data_bottom, feature_names)
