from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from DataSet_Ini import data_set_ini
from Models import print_sets, random_forest, xgboost, artificial_neural_network, support_vector_machine, ada_boost
from analysis import print_confusion_matrix, print_scores, data_analysis

'''
Togliere il commento per mostrare i risultati della parte di analisi

data_analysis('Credit_card_fraud_dataset/creditcard.csv')
'''

X, y = data_set_ini('Credit_card_fraud_dataset/creditcard.csv', 5)

scalar = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)
x_v_train, x_validate, y_v_train, y_validate = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

x_v_train = scalar.fit_transform(x_v_train)
x_validate = scalar.fit_transform(x_validate)
x_test = scalar.fit_transform(x_test)

print_sets(x_v_train, y_v_train, x_validate, y_validate, x_test, y_test)

confusion_matrix, score = random_forest(x_v_train, y_v_train, x_test, y_test, list(X.columns.values))
scores = {'Random Forest': {'Test': score}}
print_confusion_matrix(confusion_matrix)

confusion_matrix, score = xgboost(x_v_train, y_v_train, x_test, y_test)
scores['XGboost'] = {'Test': score}
print_confusion_matrix(confusion_matrix)

confusion_matrix, score = artificial_neural_network(x_v_train, y_v_train, x_validate, y_validate, x_test, y_test)
scores['ANN'] = {'Test': score}
print_confusion_matrix(confusion_matrix)

confusion_matrix, score = support_vector_machine(x_v_train, y_v_train, x_test, y_test)
scores['SVM'] = {'Test': score}
print_confusion_matrix(confusion_matrix)

confusion_matrix, score = ada_boost(x_v_train, y_v_train, x_test, y_test)
scores['ADAboost'] = {'Test': score}
print_confusion_matrix(confusion_matrix)

print_scores(scores)
