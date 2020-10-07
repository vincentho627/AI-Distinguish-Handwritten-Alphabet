import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def Train(file_name, class_name):

    file = pd.read_csv(file_name, sep=",")
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    label = LabelEncoder()

    # counts no. of 0 and 1
    file[class_name].value_counts()

    # removes :, -1 column
    X = file.drop(class_name, axis=1)
    y = file[class_name]

    # Separates training set into train and test set where test set is 20%. Setting random_state a fixed value will
    # guarantee that same sequence of random numbers are generated each time you run the code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    # transform here because we already fitted the data
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    # use stochastic gradient descent
    # verbose makes it print out the loss (cost) function per iteration
    mlp = MLPClassifier(solver="adam", hidden_layer_sizes=(512, 127, 50), max_iter=300, verbose=1)
    # if you want to try it out with logistic regression
    # mlp = LogisticRegression(solver="lbfgs", max_iter=300)

    # trains neural network
    # won't converge because of CI's time constraints, so we catch the warning and are ignore it here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(X_train, y_train)

    return mlp, X_train, y_train, X_val, y_val, X_test, y_test


def Predict(mlp, l, expected):
    prediction = mlp.predict(l)
    print(classification_report(expected, prediction))


def Print_Result(mlp: MLPClassifier, X_train, y_train, X_val, y_val, X_test, y_test):
    # predicts validation data
    predictions = mlp.predict(X_val)

    print("------------------------------------------------------------------------------")
    print("VALIDATION DATA RESULT")
    print("------------------------------------------------------------------------------")
    # compares result
    print(classification_report(y_val, predictions))
    # prints how many predicted correct and wrong, [right, wrong]
    print(confusion_matrix(y_val, predictions))
    # prints accuracy score
    print(accuracy_score(y_val, predictions))

    # predict test data
    print("------------------------------------------------------------------------------")
    print("TEST DATA RESULT")
    print("------------------------------------------------------------------------------")
    y_new = mlp.predict(X_test)
    print(classification_report(y_test, y_new))
    print(confusion_matrix(y_test, y_new))
    print(accuracy_score(y_test, y_new))

    print("------------------------------------------------------------------------------")
    print("OVERALL DATA")
    print("------------------------------------------------------------------------------")
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Validation set score: %f" % mlp.score(X_val, y_val))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    # if used logistic regression comment the following code out
    plt.plot(mlp.loss_curve_)
    plt.show()


