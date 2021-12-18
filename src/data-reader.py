from pandas import read_csv
from sklearn.model_selection import StratifiedShuffleSplit

data_raw = read_csv("./data/connect-4.csv", header=0)
# class DataFrame

data_feature = data_raw.iloc[:, :-1]
data_class = data_raw.iloc[:, -1:]

train_propotions = [0.4, 0.6, 0.8, 0.9]
datasets = []

for train_propotion in train_propotions:
    splitter = StratifiedShuffleSplit(
        train_size=train_propotion, test_size=1-train_propotion)
    splitter.split(data_feature, data_class)
    
