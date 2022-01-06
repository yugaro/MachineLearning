import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder as laE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
import IPython
auth.authenticate_user()
sns.set()
plt.style.use('ggplot')
font = {'family': 'sans-serif'}
matplotlib.rc('font', **font)

# visualization of DataFrame　(graph)

def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df)

# visualization of DataFrame (plot)


def plot_data(df):

    # extract data by gender
    df_male = df[df['gender'] == 'male']
    df_female = df[df['gender'] == 'female']

    # extract data by score
    df_score1 = df[df['score'] == 1]
    df_score2 = df[df['score'] == 2]
    df_score3 = df[df['score'] == 3]
    df_score4 = df[df['score'] == 4]

    # extract average data
    df_mean = df.mean()
    df_mean['P target(15%)'] = df_mean['P target(15%)'] * 4
    df_mean['F target(25%)'] = df_mean['F target(25%)'] * 9
    df_mean['C target(60%)'] = df_mean['C target(60%)'] * 4
    df_mean['P'] = df_mean['P'] * 4
    df_mean['F'] = df_mean['F'] * 9
    df_mean['C'] = df_mean['C'] * 4
    df_diff = pd.DataFrame([[df_mean['E'], df_mean['EER']],
                            [df_mean['P'], df_mean['P target(15%)']],
                            [df_mean['F'], df_mean['F target(25%)']],
                            [df_mean['C'], df_mean['C target(60%)']]],
                           index=['E', 'P', 'F', 'C'], columns=['individual(ave)', 'target(ave)'])

    # visualize data in the following
    fig = plt.figure(figsize=(18, 12))

    # Histogram of Score Data
    ax = fig.add_subplot(2, 3, 1)
    df['score'].plot(ax=ax, kind='hist')
    ax.set_title('Histogram of Score Data')
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of people')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['worst', 'bad', 'good', 'best'])

    # Histogram of Age (in case of gender)
    ax = fig.add_subplot(2, 3, 2)
    df_male['age'].plot(ax=ax, kind='hist', alpha=0.6,
                        color='blue', label='male')
    df_female['age'].plot(ax=ax, kind='hist', alpha=0.6,
                          color='red', label='female')
    ax.set_title('Histogram of Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of people')
    ax.legend(loc='best')

    # Scatter of Vegetable and EER (in case of score)
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(df_score1['vegetable'], df_score1['E'],
               c='red', label='worst')
    ax.scatter(df_score2['vegetable'], df_score2['E'],
               c='blue', label='bad')
    ax.scatter(df_score3['vegetable'], df_score3['E'],
               c='green', label='good')
    ax.scatter(df_score4['vegetable'], df_score4['E'],
               c='yellow', label='best')
    ax.axvline(350, ls="-.", color="darkgreen", label="target")
    ax.set_title('Scatter of Vegetable and EER')
    ax.set_xlabel('Vegetable [g]')
    ax.set_ylabel('EER [kcal]')
    ax.grid(True)
    ax.legend(loc='best')

    # Scatter of Vegetable and Carbohydrate (in case of score)
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(df_score1['vegetable'], df_score1['C'],
               c='red', label='worst')
    ax.scatter(df_score2['vegetable'], df_score2['C'],
               c='blue', label='bad')
    ax.scatter(df_score3['vegetable'], df_score3['C'],
               c='green', label='good')
    ax.scatter(df_score4['vegetable'], df_score4['C'],
               c='yellow', label='best')
    ax.axvline(350, ls="-.", color="darkgreen", label="target")
    ax.set_title('Scatter of Vegetable and Carbohydrate')
    ax.set_xlabel('Vegetable [g]')
    ax.set_ylabel('Carbohydrate [g]')
    ax.grid(True)
    ax.legend(loc='best')

    # Scatter of Vegetable and Salt (in case of score)
    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(df_score1['vegetable'], df_score1['salt'],
               c='red', label='worst')
    ax.scatter(df_score2['vegetable'], df_score2['salt'],
               c='blue', label='bad')
    ax.scatter(df_score3['vegetable'], df_score3['salt'],
               c='green', label='good')
    ax.scatter(df_score4['vegetable'], df_score4['salt'],
               c='yellow', label='best')
    ax.axvline(350, ls="-.", color="darkgreen", label="target")
    ax.axhline(8, ls="-.", color="navy", label="thres(M)")
    ax.axhline(7, ls="-.", color="magenta", label="thres(F)")
    ax.set_title('Scatter of Vegetable and Salt')
    ax.set_xlabel('Vegetable [g]')
    ax.set_ylabel('Salt [g]')
    ax.grid(True)
    ax.legend(loc='best')

    # Differences between the average of Individual and target data
    ax = fig.add_subplot(2, 3, 6)
    df_diff.plot(kind='bar', ax=ax)
    ax.set_title(
        'Differences of (E, F, P, C) \n between Individual and target data')
    ax.set_xticklabels(labels=['E', 'F', 'P', 'C'], rotation='horizontal')
    ax.set_ylabel('[kcal]')
    ax.grid(True)
    ax.legend(loc='best')

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig('sample_data_feature.png')
    plt.show()


# transform qualitative variable into quantitative variable
def label_encording(column_data):

    le = laE().fit(column_data)
    column_data = le.transform(column_data)
    return column_data


# data cleansing
def preprocess(df):

    # delete column with NAN
    df = df.copy().dropna()

    # labeling qualitative variable
    for column_name, item in df.iteritems():
        if df[column_name].dtypes == object:
            df[column_name] = label_encording(df[column_name])
    return df


# create feature
def create_feature(df):

    # differance between traget EER and indivisuals
    df['difference (EER value)'] = df['EER'] - df['E']

    # differance between traget P and indivisuals
    df['difference (P value)'] = df['P target(15%)'] - df['P']

    # differance between traget F and indivisuals
    df['difference (F value)'] = df['F target(25%)'] - df['F']

    # differance between traget C and indivisuals
    df['difference (C value)'] = df['C target(60%)'] - df['C']

    # differance between traget ratio P and indivisuals
    df['difference (P ratio)'] = 15 - df['P'] * 400 / df['E']

    # differance between traget ratio F and indivisuals
    df['difference (F ratio)'] = 25 - df['F'] * 900 / df['E']

    # differance between traget ratio C and indivisuals
    df['difference (C ratio)'] = 60 - df['C'] * 400 / df['E']

    # differances between traget amount of salt and indiviluals by gender
    df['differcence (salt)'] = df['salt'].where(
        df['gender'] == 'male', 8 - df['salt'])
    df['differcence (salt)'] = df['differcence (salt)'].where(df['gender'] == 'female',
                                                              7 - df['salt'])

    return df


def data_split(X, y):

    # split data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, y_train, X_test, y_test


# LightGBM
class LightGBM:
    def __init__(self, params=None):
        
        self.model = None
        if params is not None:
            self.params = params
        else:
            # parameter setting
            self.params = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'feature_pre_filter': False,
                'lambda_l1': 5.239619814742854e-05,
                'lambda_l2': 8.730358273072029e-08,
                'num_leaves': 15,
                'max_depth': 30,
                'feature_fraction': 0.948,
                'bagging_fraction': 1.0,
                'bagging_freq': 0,
                'learning_rate': 0.005,
            }
        self.iterations = 5000
        self.early_stopping = 100

    # model fitting
    def fit(self, X_train, y_train, X_val, y_val):

        model_lgb_train = lgb.Dataset(X_train, y_train)
        model_lgb_valid = lgb.Dataset(X_val, y_val)
        evaluation_results = {}

        # get starting time of model fitting
        ts = time.time()

        # model leaning based on train - test - valid
        self.model = lgb.train(self.params,
                               train_set=model_lgb_train,
                               valid_sets=[model_lgb_train, model_lgb_valid],
                               num_boost_round=self.iterations,
                               early_stopping_rounds=self.early_stopping,
                               valid_names=['Train', 'Valid'],
                               evals_result=evaluation_results,
                               verbose_eval=False
                               )

        # compute excution time of fitting
        evaluation_results['Excution Time [s]'] = time.time() - ts

        return evaluation_results

    # predict
    def predict(self, X_test):

        y_pred_prob = self.model.predict(
            X_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred

    # evaluate performance
    def performance_analsis(self, y_test, y_pred, evaluation_results):

        # visual Accuracy score
        accuracy = accuracy_score(y_test, y_pred)

        # visual Cohen’s Kappa score
        kappa = cohen_kappa_score(y_test, y_pred)

        # visualize Accuracy, Cohen’s Kappa, Excution Time
        display(pd.DataFrame([[accuracy, kappa, evaluation_results['Excution Time [s]']]],
                             columns=['Accuracy Score',
                                      'Kappa Score', 'Excution Time [s]'],
                             index=['Record']))

        # transition of Log Loss
        plt.figure(figsize=(6, 4))
        plt.plot(evaluation_results['Train']
                 ['multi_logloss'], label='Train Data')
        plt.plot(evaluation_results['Valid']
                 ['multi_logloss'], label='Valid Data')
        plt.xlabel('Boosting Round')
        plt.ylabel('Log Loss')
        plt.title('Transition of Log loss')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig('transition_of_log_loss.png')
        plt.show()

    # 特徴量の重要度の可視化
    def get_feature(self, df):

        # extract feature importance
        df_importance = pd.DataFrame(
            self.model.feature_importance(),
            index=df.drop(['score'], axis=1).columns, columns=['importance'])

        # rearrange feature importance in descending order
        df_importance = df_importance.sort_values(
            'importance', ascending=True)

        # normalization of feature importance
        df_importance = (df_importance - df_importance.min()) / \
            (df_importance.max() - df_importance.min())

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(df_importance)),
                 df_importance['importance'].values, align='center')
        plt.yticks(np.arange(len(df_importance)), df_importance.index)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.savefig('feature_importance.png')
        plt.show()


# model learning
def learning_model(model, df):

    # split data into target(score) and the others
    df_train_data = df.drop(['score'], axis=1).values
    df_target = df["score"].values - 1

    # split data into train - test - valid
    X_train, y_train, X_test, y_test = data_split(df_train_data, df_target)
    X_train, y_train, X_val, y_val = data_split(X_train, y_train)

    # model fitting
    evaluation_results = model.fit(X_train, y_train, X_val, y_val)
    return X_test, y_test, evaluation_results


if __name__ == '__main__':

    # read data from Google Spreadsheet
    auth.authenticate_user()
    gc = gspread.authorize(GoogleCredentials.get_application_default())
    worksheet = gc.open_by_key(
        '1ib2mEpc29e8-SHXrxZh6HeGOAexlIAL85YEwawM_WV0').sheet1
    rows = worksheet.get_all_values()
    header = rows.pop(0)
    df = pd.DataFrame.from_records(rows, columns=header, index='ID')

    # change empty data into None
    df = df.where(df != '')

    # visualization of　basic statistics
    display(df[header[1:3]].describe(include="all"))
    display(df[header[3:]].astype(float).describe().loc[[
            'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

    # visualization of data
    df_dist = pd.concat(
        [df[header[1:3]], df[header[3:]].astype(float)], axis=1)
    plot_data(df_dist)

    # data cleansing
    df_dist = preprocess(df_dist)

    # feature extraction and generation
    df_dist = create_feature(df_dist)

    # create learning model
    model_lgb = LightGBM()
    X_test, y_test, evaluation_results = learning_model(model_lgb, df_dist)

    # model predict
    y_pred = model_lgb.predict(X_test)

    # evaluation of model
    model_lgb.performance_analsis(y_test, y_pred, evaluation_results)

    # visualization of feature importance
    model_lgb.get_feature(df_dist)
