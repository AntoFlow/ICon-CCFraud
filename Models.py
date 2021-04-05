from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from xgboost import XGBClassifier
import pandas
from sklearn.tree import export_graphviz


import numpy as np

def print_score(label, prediction, train=True):
    if train:
        clf_report = pandas.DataFrame(classification_report(label, prediction, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Classification Report:\n{clf_report}")
        print("_______________________________________________")
        #print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")

    elif train == False:
        clf_report = pandas.DataFrame(classification_report(label, prediction, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Classification Report:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")
'''
Perchè questi numeri? I PARAMETRI SIGNORA, SIGNORAAAAAA
'''
def RandomForest(x_train, y_train,x_test, y_test, feature_list):
    random_forest = RandomForestClassifier(n_estimators=100, oob_score=False)
    random_forest.fit(x_train, y_train)

    y_train_pred = random_forest.predict(x_train)
    y_test_pred = random_forest.predict(x_test)

    print_score(y_train, y_train_pred, train=True)
    print_score(y_test, y_test_pred, train=False)

    estimator = random_forest.estimators_[5]
    print_tree(estimator, feature_list)

    return confusion_matrix(y_test, y_test_pred)

def XGboost(x_train, y_train, x_test, y_test):
    xg_boost = XGBClassifier(use_label_encoder=False)
    xg_boost.fit(x_train, y_train, eval_metric="aucpr")

    y_train_pred = xg_boost.predict(x_train)
    y_test_pred = xg_boost.predict(x_test)

    print_score(y_train, y_train_pred, train=True)
    print_score(y_test, y_test_pred, train=False)

def print_sets(x_v_train, y_v_train, x_validate, y_validate, x_test, y_test):
    print(f"Addestramento: \t x: {x_v_train.shape},\t y: ({len(y_v_train)}, 1)\n{'_'*52}")
    print(f"Validazione: \t x: {x_validate.shape},\t y: ({len(y_validate)}, 1)\n{'_'*52}")
    print(f"Test: \t\t\t x: {x_test.shape},\t y: ({len(y_test)}, 1)")

def print_tree(estimator, feature_list):
    export_graphviz(estimator, out_file='tree.dot', feature_names=feature_list,
            class_names='Class', rounded=True, proportion=False, precision=2, filled=True)
