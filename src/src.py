# Data preparation

## Read data
from pandas import read_csv

data_raw = read_csv("./data/connect-4.csv", header=0)
print(data_raw)


## Preprocessing Data
data_feature = data_raw.iloc[:, :-1]
data_label = data_raw.iloc[:, -1:]

feature_names = data_feature.columns
class_names = ['draw', 'loss', 'win']

x = data_feature.to_numpy()
y = data_label.to_numpy()

for i in range( 0, len( x[:] ) ):
    for j in range( 0, len( x[i][:] ) ):
        if x[i][j] == 'b':
            x[i][j] = -1
        elif x[i][j] == 'x':
            x[i][j] = 0
        elif x[i][j] == 'o':
            x[i][j] = 1
        else:
            print(x[i][j])
            raise Exception("Invalid Value")


## Train/Test Separation with Different Proportions
from sklearn.model_selection import StratifiedShuffleSplit

train_propotions = [0.4, 0.6, 0.8, 0.9]
datasets = []

for prop in train_propotions:
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=prop)
    
    for train_idx, test_idx in splitter.split(x, y):
        feature_train, label_train = x[train_idx], y[train_idx]    
        feature_test, label_test = x[test_idx], y[test_idx]
             
        datasets.append(
            {
                "feature_train" : feature_train,
                "label_train" : label_train,
                "feature_test" : feature_test,
                "label_test" : label_test
            }
        )




# Decision Tree Classifier

## Modelling and Visualization
from sklearn.tree import DecisionTreeClassifier

dec_trees = []
for dataset in datasets:
    dec_tree = DecisionTreeClassifier(criterion='entropy')
    
    dec_tree = dec_tree.fit(dataset["feature_train"], dataset["label_train"])
    
    dec_trees.append(dec_tree)

from sklearn.tree import export_graphviz
from graphviz import Source

for dec_tree in dec_trees:
    idx = dec_trees.index(dec_tree)
    corr_prop = train_propotions[idx]
    
    filename = f"tree-{corr_prop}_Full"
    
    doc_data = export_graphviz(
        dec_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True)
    
    Source(doc_data).render(filename=filename, format='svg')


## Evaluation
from numpy import concatenate
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

for dec_tree, dataset in zip(dec_trees, datasets):
    pred = dec_tree.predict(dataset["feature_test"])
    ground_truth = concatenate(dataset["label_test"])
    
    print( classification_report(ground_truth, pred, labels=class_names) )
    
    ConfusionMatrixDisplay.from_predictions(ground_truth, pred, labels=class_names)




# Evaluating Tree Depth and Accuracy
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score
from graphviz import Source
from numpy import concatenate

dataset = datasets[2]

max_depths = (None, 2, 3, 4, 5, 6, 7)
for max_depth in max_depths:
    dec_tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    dec_tree = dec_tree.fit(dataset["feature_train"], dataset["label_train"])
    
    filename = f"tree-{0.8}_Depth-{max_depth}"
    
    doc_data = export_graphviz(
        dec_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True)
    
    Source(doc_data).render(filename=filename, format='svg')
    
    pred = dec_tree.predict(dataset["feature_test"])
    ground_truth = concatenate(dataset["label_test"])
    
    print( accuracy_score(ground_truth, pred) )