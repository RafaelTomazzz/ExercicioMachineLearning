import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, \
StratifiedKFold, RepeatedStratifiedKFold

RS = 42
df = pd.read_csv("datasets/titanic/train.csv")
y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

# (a) Quantas linhas e colunas? Qual a proporção da classe positiva?

linhas = df.shape[0]
colunas = df.shape[1]
qVivos = df[df['Survived'] == 1].shape[0]
proporcao = (100 * qVivos) / linhas

# 100 - linhas
#  x - vivos
#
# x * linhas = 100 * vivos
# x = 100 * vivos / linhas

print("A \n")
print("Quantidade de linhas - " + str(linhas))
print("Quantidade de colunas - " + str(colunas))
print("Classe positiva - " + str(round(proporcao, 2)) + '%') 

# (b) Quais colunas têm valores ausentes e em que percentual?

colunasAusentes = df.isnull().sum()

print("\nB \n")
print("Colunas com valor ausente - " + str())

for coluna in colunasAusentes.index:
    if colunasAusentes[coluna] > 0:
        print(coluna)

# (c) Calcule a taxa de sobrevivência por Sex, por Pclass e pelo cruzamento das duas. Apresente em uma tabela.

totalM = df[(df['Sex'] == 'male') & (df['Survived'] == 1)].shape[0]
taxaM = (totalM * 100) / linhas

totalF = df[(df['Sex'] == 'female') & (df['Survived'] == 1)].shape[0]
taxaF = (totalF * 100) / linhas

print("\nC \n") 
print("Taxa de sobrevivência por sexo: ")
print("Masculino - " + str(round(taxaM, 2)) + '%')
print("Feminino - " + str(round(taxaF, 2)) + '%')

pclass = df['Pclass'].unique()

print("\nTaxa de sobrevivência por Pclass: ")

for classe in pclass:
    totalClasse = df[(df['Pclass'] == classe) & (df['Survived'] == 1)].shape[0]
    taxaClasse = (totalClasse * 100) / linhas
    print("Classe " + str(classe) + " - " + str(round(taxaClasse, 2)) + '%')

print("\nTaxa de sobrevivência por cruzamento de sexo e Pclass: ")

for classe in pclass:
    totalMClasse = df[(df['Pclass'] == classe) & (df['Sex'] == 'male') & (df['Survived'] == 1)].shape[0]
    taxaMClasse = (totalMClasse * 100) / linhas
    print("Classe " + str(classe) + " Masculino - " + str(round(taxaMClasse, 2)) + '%')

    totalFClasse = df[(df['Pclass'] == classe) & (df['Sex'] == 'female') & (df['Survived'] == 1)].shape[0]
    taxaFClasse = (totalFClasse * 100) / linhas
    print("Classe " + str(classe) + " Feminino - " + str(round(taxaFClasse, 2)) + '%')

# (d) Com base apenas nessas tabelas, qual regra de uma única condição você proporia? Qual
# acurácia ela atinge no conjunto completo?
