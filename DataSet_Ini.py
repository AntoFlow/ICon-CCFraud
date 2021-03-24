import pandas
import numpy as np

'''
La funzione prende in input percorso e nome file;
divide inizialmente i dati in normali e fraudolenti,
per ribilanciare il dataset sulla percentuale (normal_perc)
ed elimina la colonna "Class" e successivamente divide in x e y
x per il dataset meno il "class" e y solo la colonna "class"
la funzione ha come output x e y
'''


def data_set_ini(file_path, normal_perc):

    fraud_detection = pandas.read_csv(file_path, sep=",", index_col=None)

    '''
    Divido il dataset in due, splittando in normal (non fraudolente) e fraud (fraudolente);
    randomizzo l'ordine di normal e
    Successivamente si uniscono con una proporzionalità differente
    '''
    normal = fraud_detection[fraud_detection["Class"] == 0]
    normal = normal.iloc[np.random.permutation(len(normal))]
    fraud = fraud_detection[fraud_detection["Class"] == 1]

    if normal_perc > 100:
        normal_perc = 100
    elif normal_perc <= 0:
        normal_perc = 1
    normal_size = int((len(normal) * normal_perc) / 100)
    fraud_detection = pandas.concat([normal[:normal_size], fraud])

    '''
    fraud_detection = fraud_detection.iloc[np.random.permutation(len(fraud_detection))]
    print(fraud_detection)
    '''

    # Rimuovo la colonna contenente "Class"
    x = fraud_detection.drop("Class", axis='columns')
    y = fraud_detection["Class"]

    return x, y
