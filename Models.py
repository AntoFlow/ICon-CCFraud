from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.tree import export_graphviz
from sklearn import svm
from xgboost import XGBClassifier
from tensorflow import keras
from analysis import print_ann_result
import pandas

def random_forest(x_train, y_train,x_test, y_test, feature_list):
    random_forest = RandomForestClassifier(n_estimators=100, oob_score=False)
    random_forest.fit(x_train, y_train)

    y_pred = random_forest.predict(x_test)
    print_score(y_test, y_pred)

    estimator = random_forest.estimators_[5]
    print_tree(estimator, feature_list)

    return confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred)

def xgboost(x_train, y_train, x_test, y_test):
    xg_boost = XGBClassifier(use_label_encoder=False)
    xg_boost.fit(x_train, y_train, eval_metric="aucpr")

    y_pred = xg_boost.predict(x_test)
    print_score(y_test, y_pred)

    return confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred)

def ada_boost(x_train, y_train, x_test, y_test):
    ada = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100, learning_rate=0.8, random_state=42)
    ada.fit(x_train, y_train)

    y_pred = ada.predict(x_test)
    print_score(y_test, y_pred)

    return confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred)

def support_vector_machine(x_train, y_train, x_test, y_test):
    sv_machine = svm.SVC(kernel='linear')
    sv_machine.fit(x_train, y_train)

    y_pred = sv_machine.predict(x_test)
    print_score(y_test, y_pred)

    return confusion_matrix(y_test, y_pred), f1_score(y_test, y_pred)

def artificial_neural_network(x_v_train, y_v_train, x_validate, y_validate, x_test, y_test):
    ann = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(x_v_train.shape[-1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    metrics = [
        keras.metrics.Accuracy(name='accuracy'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]

    ann.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=metrics)
    callbacks = [keras.callbacks.ModelCheckpoint('fraud_model_at_epoch_{epoch}.h5')]

    result = ann.fit(x_v_train, y_v_train, validation_data=(x_validate, y_validate), batch_size=2048,
        epochs=300, class_weight={0: 0.0017, 1: 0.9982}, callbacks=callbacks)

    print_ann_result(result)

    score = ann.evaluate(x_test, y_test)
    print(score)

    y_pred = ann.predict(x_test)
    print_score(y_test, y_pred.round())

    return confusion_matrix(y_test, y_pred.round()), f1_score(y_test, y_pred.round())


def print_sets(x_v_train, y_v_train, x_validate, y_validate, x_test, y_test):
    print(f"Addestramento: \t x: {x_v_train.shape},\t y: ({len(y_v_train)}, 1)\n{'_'*52}")
    print(f"Validazione: \t x: {x_validate.shape},\t y: ({len(y_validate)}, 1)\n{'_'*52}")
    print(f"Test: \t\t\t x: {x_test.shape},\t y: ({len(y_test)}, 1)")

def print_tree(estimator, feature_list):
    export_graphviz(estimator, out_file='tree.dot', feature_names=feature_list,
            class_names='Class', rounded=True, proportion=False, precision=2, filled=True)

def print_score(label, prediction):
    report = pandas.DataFrame(classification_report(label, prediction, output_dict=True))
    print("\nTest Result:\n_______________________________________________")
    print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
    print(f"\nClassification Report:\n{report}")
