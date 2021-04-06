import pandas
import matplotlib.pyplot as plt
import seaborn

def print_confusion_matrix(matrix):
    df_cm = pandas.DataFrame(matrix, index=["Normal", "Fraud"], columns=["Normal", "Fraud"])
    plt.figure(figsize=(10, 7))
    seaborn.heatmap(df_cm, annot=True)
    plt.show()

def data_analysis(file_path):
    data = pandas.read_csv(file_path, sep=",", index_col=None)

    LABELS = ["Normal", "Fraud"]

    # Imposto il formato a massimo due decimali
    pandas.set_option("display.float", "{:.2f}".format)

    count_classes = pandas.value_counts(data['Class'], sort=True)
    count_classes.plot(kind='bar', rot=0)
    plt.title("Distribuzione delle classi di transazione")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Classi")
    plt.ylabel("Frequenza")

    data.Class.value_counts()

    fraud = data[data['Class'] == 1]
    normal = data[data['Class'] == 0]

    print("---------------------------------------------------------")
    # Osservo la grandezza dei due gruppi
    print(f"Grandezza transizzioni fraudolente: {fraud.shape}")
    print(f"Grandezza transizzioni non-fraudolente: {normal.shape}")
    print("---------------------------------------------------------")

    # Osservo la diversa quantità di denaro utilizzata nelle classi di transazioni
    print(pandas.concat([fraud.Amount.describe(), normal.Amount.describe()], axis=1))
    print("---------------------------------------------------------")
    # Osservo se le transazioni fraudolente si verificano più di frequente in un determinato periodo di tempo
    print(pandas.concat([fraud.Time.describe(), normal.Time.describe()], axis=1))
    print("---------------------------------------------------------")

    # Traccia la funzione tempo
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title('Distribuzione Tempo(s)')

    seaborn.distplot(data['Time'], color='blue')

    # Traccia la funzione importo
    plt.subplot(2, 1, 2)
    plt.title('Distribuzione delle quantità')
    seaborn.distplot(data['Amount'], color='blue')

    # uso la heatmap per trovare le relazioni più marcate
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(data=data.corr(), cmap="seismic")

    plt.show()

'''
Procedura di stampa del grafico che evidenzia l'evoluzione delle metriche durante l'addestramento della rete.
'''
def print_ann_result(result):
    plt.figure(figsize=(12, 16))

    plt.subplot(3, 2, 1)
    plt.plot(result.history['loss'], label='Loss')
    plt.plot(result.history['val_loss'], label='val_Loss')
    plt.title('Loss Function evolution during training')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(result.history['fn'], label='fn')
    plt.plot(result.history['val_fn'], label='val_fn')
    plt.title('Accuracy evolution during training')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(result.history['precision'], label='precision')
    plt.plot(result.history['val_precision'], label='val_precision')
    plt.title('Precision evolution during training')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(result.history['recall'], label='recall')
    plt.plot(result.history['val_recall'], label='val_recall')
    plt.title('Recall evolution during training')
    plt.legend()


'''
Procedura di stampa del grafico di confronto di modelli.
'''
def print_scores(scores):
    score_data = pandas.DataFrame(scores)
    score_data.plot(kind='barh', figsize=(15, 6))
    plt.show()

