import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, \
StratifiedKFold, RepeatedStratifiedKFold

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

# shuffle garante que a proporção de classes seja mantida em cada fold, mesmo que os dados estejam ordenados de alguma forma.
# random_state garante que a divisão seja reproduzível, ou seja, que a mesma divisão seja obtida em diferentes execuções do código.

for train_index, test_index in skf.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]