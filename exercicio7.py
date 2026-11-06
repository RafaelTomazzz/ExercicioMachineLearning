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

print("Quantidade de linhas - " + str(linhas))
print("Quantidade de colunas - " + str(colunas))
print("Classe positiva - " + str(round(proporcao, 2)) + '%') 

# (b) Quais colunas têm valores ausentes e em que percentual?
# (c) Calcule a taxa de sobrevivência por Sex, por Pclass e pelo cruzamento das duas. Apresente em uma tabela.
# (d) Com base apenas nessas tabelas, qual regra de uma única condição você proporia? Qual
# acurácia ela atinge no conjunto completo?
