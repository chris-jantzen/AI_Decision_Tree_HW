from pandas import read_excel
from random import sample

table = read_excel("P4Data.xlsx")

data_sets = []
for x, y, c in zip(table['X'], table['Y'], table['Class']):
    data_sets.append([x, y, c])

training_indices = sample(range(len(table)), int(len(table)*.70))
