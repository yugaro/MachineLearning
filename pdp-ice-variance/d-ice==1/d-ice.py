import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
np.random.seed(0)
plt.rcParams['font.size'] = 18


def data_cleansing(dataset):
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    feature_names = [X.columns[7], X.columns[20], X.columns[21],
                     X.columns[24], X.columns[27], X.columns[28]]
    mm = MinMaxScaler()
    X_train = mm.fit_transform(X[feature_names])
    y_train = dataset.target
    return X_train, y_train, feature_names


def derivative_func(prior_data, post_data, h):
    # # compute derivative
    return (post_data - prior_data) / h


def plot_analysis(grid_value, var_dice_data, feature_names):

    # plot data
    var_max = np.max(var_dice_data)

    feature_nums = [1, 2]
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(wspace=0.2)
    for k in feature_nums:
        ax = fig.add_subplot(1, len(feature_nums), k)
        ax.plot(grid_value[k][: -1], var_dice_data[k], color='c', label='Var. d-ICE plot')
        ax.set_xlabel(feature_names[k], fontsize=18)
        ax.set_ylabel('var. d-ICE', fontsize=18)
        ax.set_ylim(-20, var_max + 20)
        ax.grid(linestyle='dotted', lw=0.5)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
    fig.suptitle("Feature - Variance of d-ICE")
    fig.savefig('./images/var_dice_ex.png')


# def plot_analysis2(grid_value, predicted_dvar, predicted_dmean, feature_names):
#     # # plot derivative pdp with the variance of derivative of ice

#     # preparing
#     grid_value = grid_value[: -1]
#     predicted_std = np.sqrt(predicted_dvar)

#     # plot data
#     fig = plt.figure(figsize=(18, 9))
#     fig.subplots_adjust(hspace=0.5, wspace=0.5)
#     for k in range(len(feature_names)):
#         ax = fig.add_subplot(
#             2, len(feature_names) / 2 + len(feature_names) % 2, k + 1)
#         ax.plot(grid_value, predicted_dmean[k], color='blue', label='mean')
#         ax.fill_between(grid_value, predicted_dmean[k] + 2 * predicted_std[k],
#                         predicted_dmean[k] - 2 * predicted_std[k], alpha=.2, color='blue', label='std')
#         ax.set_xlabel(feature_names[k], fontsize=18)
#         ax.set_ylabel('deriv. pdp (benign prob.)', fontsize=18)
#         ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
#                   borderaxespad=0, fontsize=15)
#     fig.suptitle("feature - deriv. pdp (benign prob.)")
#     fig.savefig('./images/test02.pdf')


def plot_analysis3(grid_value, pdp_data, dpdp_data, var_dice_data, feature_names):

    # plot data
    pdp_data_upper = np.zeros((pdp_data.shape[0], pdp_data.shape[1]))
    pdp_data_bottom = np.zeros((pdp_data.shape[0], pdp_data.shape[1]))
    pdp_data_upper[:, 0] = pdp_data[:, 0]
    pdp_data_bottom[:, 0] = pdp_data[:, 0]

    h = grid_value[:, 1] - grid_value[:, 0]
    std_dice_data = np.sqrt(var_dice_data)

    # print(h)
    # print(h * (dpdp_data[:, 0] + 2 * std_dice_data[:, 0]))

    for i in range(pdp_data.shape[1] - 1):
        pdp_data_upper[:, i + 1] = pdp_data[:, i] + h * \
            (dpdp_data[:, i] + 2 * std_dice_data[:, i])
        pdp_data_bottom[:, i + 1] = pdp_data[:, i] + h * \
            (dpdp_data[:, i] - 2 * std_dice_data[:, i])

    feature_nums = [1, 2]
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(wspace=0.2)
    for k in feature_nums:
        ax = fig.add_subplot(1, len(feature_nums), k)

        ax.plot(grid_value[k], pdp_data[k], color='red', label='PDP')
        ax.fill_between(grid_value[k], pdp_data_upper[k], pdp_data_bottom[k], color='c', alpha=.3, label='STD')
        ax.set_xlabel(feature_names[k], fontsize=18)
        ax.set_ylabel('benign prob.', fontsize=18)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
        ax.grid(linestyle='dotted', lw=0.5)
    fig.suptitle("Feature - Benign Prob. (with STD.)")
    fig.savefig('./images/pdp_std_ex.png')


class Var_dIce:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        # # model fitting
        self.model.fit(X_train, y_train)

    def analyze_var_dice(self, X_train, grid_resolution):
        # # analyze variances of derivative of ice
        grid_value = np.zeros((X_train.shape[1], grid_resolution))
        pdp_data = np.zeros((X_train.shape[1], grid_resolution))
        ice_data = np.zeros((X_train.shape[1], X_train.shape[0], grid_resolution))
        dpdp_data = np.zeros((X_train.shape[1], grid_resolution - 1))
        dice_data = np.zeros((X_train.shape[1], X_train.shape[0], grid_resolution - 1))
        var_dice_data = np.zeros((X_train.shape[1], grid_resolution - 1))

        for k in range(X_train.shape[1]):
            # get the trajectories of ice for each features
            pd_data = partial_dependence(
                self.model, X_train, features=[k], percentiles=(0, 1), grid_resolution=grid_resolution, kind='both')

            # get values of grid and data of ice
            grid_value[k] = np.array(pd_data['values'][0])
            pdp_data[k] = pd_data['average'][0]
            ice_data[k] = pd_data['individual'][0]

            # get gaps between grid
            h = grid_value[k][1] - grid_value[k][0]
            for i in range(ice_data[k].shape[1] - 1):
                # compute the derivative of ice based on derivative func
                dpdp_data[k][i] = derivative_func(
                    pdp_data[k][i], pdp_data[k][i + 1], h)
                dice_data[k][:, i] = derivative_func(
                    ice_data[k][:, i], ice_data[k][:, i + 1], h)
            var_dice_data[k] = np.var(dice_data[k], axis=0)

        return grid_value, pdp_data, ice_data, dpdp_data, dice_data, var_dice_data


if __name__ == '__main__':
    # get dataset and implement cleansing
    dataset = load_breast_cancer()
    X_train, y_train, feature_names = data_cleansing(dataset)
    print(feature_names)

    # fit model
    rfc = RandomForestClassifier()
    vdi = Var_dIce(model=rfc)
    vdi.fit(X_train, y_train)

    # set grid resulution and get grid values and prediction variance
    grid_resolution = 200
    grid_value, pdp_data, ice_data, dpdp_data, dice_data, var_dice_data = vdi.analyze_var_dice(
        X_train=X_train, grid_resolution=grid_resolution)

    # plot data
    plot_analysis(grid_value, var_dice_data, feature_names)

    # plot data2
    # plot_analysis2(grid_value, predicted_dvar, predicted_dmean, feature_names)

    # plot data3
    plot_analysis3(grid_value, pdp_data,
                   dpdp_data, var_dice_data, feature_names)

    feature_nums = [1, 2]
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(wspace=0.2)
    for k in feature_nums:
        ax = fig.add_subplot(1, len(feature_nums), k)
        ax.plot(grid_value[k], pdp_data[k], color='red', label='PDP')
        ax.set_xlabel(feature_names[k], fontsize=18)
        ax.set_ylabel('benign prob.', fontsize=18)
        ax.set_ylim(0, 1)
        # ax.set_ylim(-20, var_max + 20)
        ax.grid(linestyle='dotted', lw=0.5)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
    fig.suptitle("Feature - Benign Prob.")
    fig.savefig('./images/pdp_ex.png')

    feature_nums = [1, 2]
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(wspace=0.2)
    for k in feature_nums:
        ax = fig.add_subplot(1, len(feature_nums), k)
        ax.plot(grid_value[k], pdp_data[k], color='red', label='PDP')
        for i in range(ice_data[k].shape[0]):
            if i == 0:
                ax.plot(grid_value[k], ice_data[k][i],
                        color='turquoise', label='ICE plot', zorder=-1, alpha=.5)
            else:
                r = np.random.randn()
                if r > 0.5:
                    ax.plot(grid_value[k], ice_data[k][i],
                            color='turquoise', zorder=-1, alpha=.5)
        ax.set_xlabel(feature_names[k], fontsize=18)
        ax.set_ylabel('benign prob.', fontsize=18)
        ax.set_ylim(0, 1)
        # ax.set_ylim(-20, var_max + 20)
        ax.grid(linestyle='dotted', lw=0.5)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
                  borderaxespad=0, fontsize=15)
    fig.suptitle("Feature - Benign Prob. (with ICE)")
    fig.savefig('./images/pdp_ice_ex.png')

    # fig = plt.figure(figsize=(18, 9))
    # fig.subplots_adjust(hspace=0.1, wspace=0.2)
    # for k in range(len(feature_names)):
    #     if k == 0 or k == 1:
    #         ax = fig.add_subplot(2, len(feature_names) /
    #                              2 + len(feature_names) % 2, k + 1)
    #         ax.set_xlabel(feature_names[k], fontsize=18)
    #         ax.plot(grid_value[k], pdp_data[k],
    #                 color='red', label='PDP', linewidth=2, zorder=1)
    #         if k == 0:
    #             ax.set_ylabel('benign prob.', fontsize=18)
    #         # if k == 0:
    #         #     for i in range(predicted_point1.shape[0]):
    #         #         if i == 0:
    #         #             ax.plot(
    #         #                 grid_value, predicted_point1[i], color='blue', label='ICE', alpha=.5, linewidth=1, zorder=-0.1)
    #         #         else:
    #         #             r = np.random.randn()
    #         #             if r > 0.5:
    #         #                 ax.plot(
    #         #                     grid_value, predicted_point1[i], color='blue', alpha=.5, linewidth=1, zorder=-0.1)
    #         #     ax.set_ylabel('benign prob.', fontsize=18)
    #         # if k == 1:
    #         #     for i in range(predicted_point2.shape[0]):
    #         #         if i == 0:
    #         #             ax.plot(
    #         #                 grid_value, predicted_point2[i], color='blue', label='ICE', alpha=.5, linewidth=1, zorder=-0.1)
    #         #         else:
    #         #             r = np.random.randn()
    #         #             if r > 0.5:
    #         #                 ax.plot(
    #         #                     grid_value, predicted_point2[i], color='blue', alpha=.5, linewidth=1, zorder=-0.1)
    #         ax.set_ylim(0, 1)
    #         ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
    #                   borderaxespad=0, fontsize=15)
    #         ax.grid(linestyle='dotted', lw=0.5)
    # fig.savefig('./test04.png')
