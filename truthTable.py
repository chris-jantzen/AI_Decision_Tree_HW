from pandas import read_excel
from random import sample


def read_data():
    table = read_excel("P4Data.xlsx")
    data_sets = []
    for x, y, c in zip(table['X'], table['Y'], table['Class']):
        data_sets.append([x, y, c])
    return data_sets


def get_testing_indicies(training_indices, data_size):
    testing_indicies = []
    for index in range(data_size):
        if index not in training_indices:
            testing_indicies.append(index)
    return testing_indicies


def main():
    data_sets = read_data()

    # Randomly pick 70% of data for training
    training_indices = sample(range(len(table)), int(len(table)*.70))
    testing_indices = get_testing_indicies(training_indices, len(table))


if __name__ == "__main__":
    main()
