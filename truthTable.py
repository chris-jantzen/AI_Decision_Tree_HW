from pandas import read_excel
from random import sample


def read_data():
    table = read_excel("P4Data.xlsx")
    data_sets = []
    for x, y, c in zip(table['X'], table['Y'], table['Class']):
        data_sets.append([x, y, c])
    return data_sets


def main():
    data_sets = read_data()
    # Randomly pick 70% of data for training
    training_indices = sample(range(len(table)), int(len(table)*.70))


if __name__ == "__main__":
    main()
