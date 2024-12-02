import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

url = "https://www.openml.org/data/get_csv/1795974/phpkIxskf.csv"
data = pd.read_csv(url)

print("Информация о данных:")
print(data.info())

print("\nПервые 5 строк данных:")
print(data.head())

print("\nРаспределение целевого признака (Class):")
print(data["Class"].value_counts())

print("\nПроверка пропущенных значений:")
print(data.isnull().sum())

categorical_cols = data.select_dtypes(include=["object"]).columns

encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

scaler = StandardScaler()
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

X = data.drop("Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\nЛучшие параметры модели:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC Score:")
print(roc_auc_score(y_test, y_pred_proba))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок")
plt.xlabel("Предсказанные значения")
plt.ylabel("Истинные значения")
plt.show()

new_data = X_test.iloc[:5]  # Пример: первые 5 строк из тестовой выборки
new_predictions = best_model.predict(new_data)
print("\nРезультаты классификации для новых данных:")
print(new_predictions)
