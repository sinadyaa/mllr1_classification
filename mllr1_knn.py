from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X_train_path = r"D:\!учеба\ML\lr1\heart_disease_xtrain_bal.csv"
y_train_path = r"D:\!учеба\ML\lr1\heart_disease_ytrain_bal.csv"
X_test_path = r"D:\!учеба\ML\lr1\heart_disease_xtest.csv"
y_test_path = r"D:\!учеба\ML\lr1\heart_disease_ytest.csv"

# Загрузка
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()  # превращаем в Series
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

param_grid = {
    'n_neighbors': [3, 7, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Лучшие параметры:", grid_search.best_params_)
####################################################
# Используем лучшую модель
best_knn = grid_search.best_estimator_

# Предсказания на тестовой выборке
y_pred = best_knn.predict(X_test)

# --- Метрики ---
print("\nТочность модели (Accuracy):", accuracy_score(y_test, y_pred))
print("\nОтчёт по классам:\n", classification_report(y_test, y_pred))

# --- Матрица ошибок ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation='horizontal')
plt.title("Confusion Matrix — KNN")
plt.show()