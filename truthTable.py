from pandas import read_excel
from random import sample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def read_data():
    table = read_excel("P4Data.xlsx")
    data_sets = []
    classes = []
    for x, y, c in zip(table['X'], table['Y'], table['Class']):
        data_sets.append([x, y])
        classes.append([c])
    return (data_sets, classes)


def get_testing_indicies(training_indices, data_size):
    testing_indicies = []
    for index in range(data_size):
        if index not in training_indices:
            testing_indicies.append(index)
    return testing_indicies


def build_50_by_50():
    lst = []
    for i in range(1, 51):
        for j in range(1, 51):
            lst.append([i, j])
    return lst


def create_graph(coords, pred):
    pred0x = []
    pred0y = []
    pred1x = []
    pred1y = []
    for i in range(len(pred)):
        if pred[i] == 0:
            pred0x.append(coords[i][0])
            pred0y.append(coords[i][1])
        else:
            pred1x.append(coords[i][0])
            pred1y.append(coords[i][1])

    plt.plot(pred0x, pred0y, 'rs', pred1x, pred1y, 'bs')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 50, 0, 50])
    plt.show()


def main():
    (data_sets, classes) = read_data()

    x_train, x_test, y_train, y_test = train_test_split(
        data_sets, classes, test_size=.30)

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    n_test = build_50_by_50()
    n_pred = classifier.predict(n_test)

    create_graph(n_test, n_pred)


if __name__ == "__main__":
    main()
