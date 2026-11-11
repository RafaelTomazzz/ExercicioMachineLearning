# Treine uma LogisticRegression dentro de um Pipeline completo (imputação + padronização + one-hot). 
# Faça train_test_split com test_size=0.2 para random_state ∈ {0, 1, . . . , 29}
# e registre a acurácia de teste em cada caso.
# (a) Plote o histograma das 30 acurácias. Reporte mínimo, máximo, média e desvio-padrão.
# (b) Compare a amplitude observada com a média ± desvio obtida por RepeatedStratifiedKFold(
# n_splits=5, n_repeats=6).
# (c) Responda: se dois modelos diferem em 1,5 ponto percentual em um único holdout, é
# possível afirmar que um é melhor? Justifique com os seus números.

# Pipeline - sequência de etapas em que a saída de uma etapa vira a entrada da próxima

# LogisticRegression é uma classe da biblioteca scikit-learn que implementa um modelo de classificação. 
# Apesar do nome “regressão”, ela não prevê valores contínuos (como preço), e sim 
# categorias (como sim/não, spam/não spam, aprovado/reprovado).

# Como funciona, de forma simples

# O modelo calcula uma pontuação a partir das características (features) dos dados e passa essa pontuação por 
# uma função chamada sigmoide, que transforma qualquer número em um valor entre 0 e 1. 
# Esse valor é interpretado como probabilidade:

# perto de 1 → classe positiva
# perto de 0 → classe negativa
# o limite padrão é 0,5

# 1. Imputação
# Preencher valores que estão faltando (nulos/NaN). Modelos de ML em geral não aceitam dados vazios, então você 
# substitui por algum valor razoável:

# colunas numéricas: média ou mediana
# colunas de texto/categorias: o valor mais frequente (moda)

# 2. Padronização
# Colocar as colunas numéricas na mesma escala, com média 0 e desvio padrão 1. Sem isso, uma coluna como 
# “salário” (milhares) pode dominar uma como “idade” (dezenas) só por ter números maiores. Modelos como regressão 
# logística se beneficiam bastante.

# 3. One-hot encoding
# Transformar colunas de categorias (texto) em números, criando uma coluna 0/1 para cada categoria. Exemplo:

# train_test_split - separa a base em treino e teste. O modelo é treinado com os dados de treino e avaliado com os 
# dados de teste, que ele nunca viu antes. Isso dá uma ideia de como ele vai se sair no mundo real.

import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, \
StratifiedKFold, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


RS = 42
df = pd.read_csv("datasets/titanic/train.csv")
y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)

prep_num = Pipeline([
    ("imputação", SimpleImputer(strategy="median")),
    ("padronização", StandardScaler())
])

prep_cat = Pipeline([
    ('imputação', SimpleImputer(strategy="most_frequent")),
    ('one-hot', OneHotEncoder(handle_unknown='ignore'))
])

prep_data = ColumnTransformer([
    ('num', prep_num, ['Age', 'SibSp', 'Parch', 'Fare']),
    ('cat', prep_cat, ['Pclass', 'Sex', 'Embarked'])
])

modelo = Pipeline([
    ('preparação', prep_data),
    ('classificador', LogisticRegression(random_state=RS))
])

modelo.fit(X_treino, y_treino)
predict = modelo.predict(X_teste)

print("Acurácia do modelo:", modelo.score(X_teste, y_teste))

# (a) Plote o histograma das 30 acurácias. Reporte mínimo, máximo, média e desvio-padrão.

acuracias = []

max = {'acc': 0, 'random_state': 0}
min = {'acc': 0, 'random_state': 0}

for random_state in range(30):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    random_modelo = Pipeline([
        ('preparação', prep_data),
        ('classificador', LogisticRegression(random_state=random_state))
    ])

    random_modelo.fit(X_treino, y_treino)
    acuracia = random_modelo.score(X_teste, y_teste)

    if acuracia > max['acc']:
        max['acc'] = acuracia
        max['random_state'] = random_state

    if acuracia < min['acc'] or min['acc'] == 0:
        min['acc'] = acuracia
        min['random_state'] = random_state

    acuracias.append(acuracia)

plt.hist(acuracias, bins=10, edgecolor="black")
plt.xlabel("Acurácia")
plt.ylabel("Frequência")
plt.title("Distribuição das 30 acurácias")

print('A')

print("Mínimo:", round(np.min(acuracias), 4))
print("Máximo:", round(np.max(acuracias), 4))
print("Média:", round(np.mean(acuracias), 4))
print("Desvio-padrão:", round(np.std(acuracias), 4))

print("Melhor acurácia:", round(max['acc'], 4), "com random_state =", max['random_state'])
print("Pior acurácia:", round(min['acc'], 4), "com random_state =", min['random_state'])

plt.show()

# (b) Compare a amplitude observada com a média ± desvio obtida por RepeatedStratifiedKFold(
# n_splits=5, n_repeats=6).

print("\nB")

for random_state in range(30):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    modelo_repeated_k_fold = Pipeline([
        ('preparação', prep_data),
        ('classificador', RepeatedStratifiedKFold(random_state=random_state))
    ])

# (c) Responda: se dois modelos diferem em 1,5 ponto percentual em um único holdout, é
# possível afirmar que um é melhor? Justifique com os seus números.