from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from DataSet_Ini import data_set_ini
from Models import print_sets, RandomForest, XGboost
from analysis import print_confusion_matrix

X, y = data_set_ini('Credit_card_fraud_dataset/creditcard.csv', 1)

scalar = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)
x_v_train, x_validate, y_v_train, y_validate = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

x_v_train = scalar.fit_transform(x_v_train)
x_validate = scalar.fit_transform(x_validate)
x_test = scalar.fit_transform(x_test)

print_sets(x_v_train, y_v_train, x_validate, y_validate, x_test, y_test)
print_confusion_matrix(RandomForest(x_train, y_train, x_test, y_test, list(X.columns.values)))
#print_confusion_matrix(XGboost(x_train, y_train, x_test, y_test))
#matrix = [[830, 0], [68, 73]]
#print_confusion_matrix(matrix)
