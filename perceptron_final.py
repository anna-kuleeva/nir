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

def main():
    # how much neurons we need?
    Q1 = Q2 = 12

    hidden = [11, 11]
    loss = []
    my_dict = {'loss': 1, 'layer1': 0, 'layer2': 0}
    X = pd.read_csv('nir.csv', sep=',', usecols=[1, 2, 3, 4, 5])
    Y = pd.read_csv('nir.csv', sep=',', usecols=[6])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
    # test_data = pd.read_csv('validate.csv', sep=',')
    X_train = Normalizer().fit_transform(X_train)
    X_test = Normalizer().fit_transform(X_test)
    # create the model
    # for i in range(Q1):
        # for j in range(Q2):
    print("start training...")
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver='adam',
            max_iter=1000, shuffle=True)
    # print("neurons in hidden layer ", i, " ", j)
    clf.fit(X_train, Y_train)
    print("start validating...")
    y_pred = clf.predict(X_test)
    loss_i = zero_one_loss(Y_test, y_pred)
    # if loss_i < my_dict['loss']:
    #     my_dict['loss'] = loss_i
    #     my_dict['layer1'] = i
    #     my_dict['layer2'] = j
    # loss.append(loss_i)
    print("loss = ", loss_i)
    print("mean test accuracy = ", clf.score(X_test, Y_test))  # Return the mean accuracy on the given test data and labels.
    print("confusion matrix:\n", confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))

    #roc-auc
    print("AUC = ", roc_auc_score(Y_test, y_pred))
    RocCurveDisplay.from_predictions(Y_test, y_pred)
    plt.show()
    # plot the loss
    # print(my_dict.items())
    # plt.plot(loss)
    # plt.xlabel('neurons')
    # plt.title('loss with different amount of hidden neurons')
    # plt.show()


if __name__ == '__main__':
    main()
