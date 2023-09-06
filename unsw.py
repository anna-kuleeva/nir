import pandas as pd
from math import sqrt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import *
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def read_file():
    df = pd.read_csv('UNSW_NB15_testing-set.csv', sep=',', nrows=20000, usecols=[27, 28, 5, 6, 10, 11, 44])
    print(df.info())
    # balance
    print(df['label'].value_counts())
    rat = len(df.loc[df['label'] == 0]) // len(df.loc[df['label'] == 1])
    if rat == 0:
        rat = len(df.loc[df['label'] == 1]) // len(df.loc[df['label'] == 0])
        df_1 = df.loc[df['label'] == 0]
        df_1 = df_1.loc[df_1.index.repeat(rat)]
        df_n = pd.concat([df.loc[df['label'] == 1], df_1]).sample(frac=1)
        print(df_n['label'].value_counts())
        df_n.to_csv(path_or_buf="unsw.csv")
    else:
        df_1 = df.loc[df['label'] == 1]
        df_1 = df_1.loc[df_1.index.repeat(rat)]
        df_n = pd.concat([df.loc[df['label'] == 0], df_1]).sample(frac=1)
    print(df_n['label'].value_counts())
    df_n.to_csv(path_or_buf="unsw.csv")


def main():
   # how much neurons we need?
    Q1 = Q2 = 12

    hidden = [Q1, Q2]
    loss = []
    X = pd.read_csv('unsw.csv', sep=',', usecols=[1, 2, 3, 4, 5, 6])
    Y = pd.read_csv('unsw.csv', sep=',', usecols=[7])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5)
    # test_data = pd.read_csv('validate.csv', sep=',')
    X_train = Normalizer().fit_transform(X_train)
    X_test = Normalizer().fit_transform(X_test)
    # create the model
    for i in range(Q1):
        for j in range(Q2):
            print("start training...")
            clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver='adam',
                            max_iter=1000, shuffle=True)

        # distributions = dict(max_iter=[400, 500, 1000, 10000], beta_1=[0.8, 0.9, 0.999, 0.9999],
        #                     beta_2=[0.999, 0.9, 0.9999], epsilon=[1e-7, 1e-8, 1e-9])
        # search = RandomizedSearchCV(clf, distributions, verbose=5)
        # search.fit(X_train, Y_train)
        # print("Best parameters: ", search.best_params_)
        # print("Best score: ", search.best_score_)
        # # rand_st = []
        # for i in range(100):
        #     rand_st.append(i)
        # param_grid = {'random_state': rand_st}
        # grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=5)
        # grid_search.fit(X_train, Y_train)
        # print("Best parameters: ", grid_search.best_params_)
        # print("Best score: ", grid_search.best_score_)

            print("neurons in hidden layer ", i, " ", j)
            clf.fit(X_train, Y_train)
            print("start validating...")
            y_pred = clf.predict(X_test)
            loss_i = zero_one_loss(Y_test, y_pred)
            loss.append(loss_i)
            print("loss = ", loss_i)
            print("mean test accuracy = ", clf.score(X_test, Y_test))  # Return the mean accuracy on the given test data and labels.
            print("confusion matrix:\n", confusion_matrix(Y_test, y_pred))
            print(classification_report(Y_test, y_pred))

        #roc-auc
            print("AUC = ", roc_auc_score(Y_test, y_pred))
    # RocCurveDisplay.from_predictions(Y_test, y_pred)
    # plt.show()
    # plot the loss
    plt.plot(loss)
    plt.xlabel('neurons')
    plt.title('loss with different amount of hidden neurons')
    plt.show()


if __name__ == '__main__':
#  read_file()
    main()
