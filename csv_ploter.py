# Import the necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import csv

#Aqui você vai abrir seu arquivos
csv1 = open('v3.csv', 'r')

#Para ler os arquivos csv
leitor_csv1 = csv.reader(csv1, delimiter=';')#nao esqueça de colocar qual o tipo de delimitador entre cada célula se é , ; :

epochs = []
for i in range(10, 300, 10):
    epochs.append(i)

pos = []
neg = []

for linha in leitor_csv1:
    if linha[0] == 'modelo':
        continue
    if linha[0] == 'positive':
        for i in range(1, len(linha)-1):
            pos.append(float(linha[i]))
    elif linha[0] == 'negative':
        for i in range(1, len(linha)):
            neg.append(float(linha[i]))

plt.plot(epochs, pos, label="Positivo")
plt.plot(epochs, neg, label="Negativo")
plt.legend()
plt.title("Modelo Lucas V3")
plt.show()
