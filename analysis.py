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

#data_analysis('Credit_card_fraud_dataset/creditcard.csv')
