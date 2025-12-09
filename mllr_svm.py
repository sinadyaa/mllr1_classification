import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Пути к данным
X_train_path = r"D:\!учеба\ML\lr1\heart_disease_xtrain_bal.csv"
y_train_path = r"D:\!учеба\ML\lr1\heart_disease_ytrain_bal.csv"
X_test_path = r"D:\!учеба\ML\lr1\heart_disease_xtest.csv"
y_test_path = r"D:\!учеба\ML\lr1\heart_disease_ytest.csv"

# Загрузка
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

# --- Настройка сетки параметров ---
param_grid = {
    'C': [1],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

svm = SVC(random_state=42)

grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)

# --- Лучшая модель ---
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# --- Метрики ---
print("\n Точность модели (Accuracy):", accuracy_score(y_test, y_pred))
print("\n Отчет по классам:\n", classification_report(y_test, y_pred))

# --- Матрица ошибок ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation='horizontal') 
plt.title("Confusion Matrix — SVM")
plt.show()

# --- Важность признаков (для линейного ядра) ---
if grid_search.best_params_['kernel'] == 'linear':
    coef = best_svm.coef_[0]
    feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(coef)})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'])
    plt.gca().invert_yaxis()
    plt.title('Feature Importance (SVM — Linear Kernel)')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.show()
else:
    print("\nℹ️ Для нелинейного ядра (rbf) визуализация важности признаков недоступна.")
