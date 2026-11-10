import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, \
StratifiedKFold, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier

RS = 42
df = pd.read_csv("datasets/titanic/train.csv")
y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

# Implemente e avalie, com StratifiedKFold(n_splits=5): (i) DummyClassifier(strategy=
# "most_frequent"); (ii) DummyClassifier(strategy="stratified"); (iii) a regra de domínio
# «mulheres e crianças (Age < 12) sobrevivem». Reporte acurácia e F1 (média ± desvio) em uma
# tabela. Comente em duas frases.

# StratifiedKFold é um tipo de validação cruzada (do scikit-learn) que divide os dados em k partes (folds) mantendo, 
# em cada parte, a mesma proporção de classes que existe no dataset original.

X.dropna()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

resultados = {
    "most_frequent": {"acc": []},
    "stratified":    {"acc": []},
    "regra_dominio": {"acc": []},
}

# shuffle garante que a proporção de classes seja mantida em cada fold, mesmo que os dados estejam ordenados de alguma forma.
# random_state garante que a divisão seja reproduzível, ou seja, que a mesma divisão seja obtida em diferentes execuções do código.

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dummy_most_frequent = DummyClassifier(strategy="most_frequent", random_state=RS)
    dummy_most_frequent.fit(X_train, y_train)
    pred_most_frequent = dummy_most_frequent.predict(X_test)
    resultados["most_frequent"]["acc"].append(dummy_most_frequent.score(X_test, y_test))

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dummy_stratified = DummyClassifier(strategy="stratified", random_state=RS)
    dummy_stratified.fit(X_train, y_train)
    pred_stratified = dummy_stratified.predict(X_test)
    resultados["stratified"]["acc"].append(dummy_stratified.score(X_test, y_test))


print("Resultados do DummyClassifier(strategy='most_frequent'):")
print("Acurácia média:", np.mean(resultados["most_frequent"]["acc"]))

# Estratificado (stratified) significa que a divisão ou o sorteio respeita a proporção das classes do alvo.
print("Resultados do DummyClassifier(strategy='stratified'):")
print("Acurácia média:", np.mean(resultados["stratified"]["acc"]))