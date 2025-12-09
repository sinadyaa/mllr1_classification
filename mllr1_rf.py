import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


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
    'n_estimators': [100, 200],       # Количество деревьев
    'max_depth': [10, 20, None],      # Глубина деревьев
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# --- Метрики ---
print("\n Точность модели (Accuracy):", accuracy_score(y_test, y_pred))
print("\n Отчет по классам:\n", classification_report(y_test, y_pred))

# --- Матрица ошибок ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation='horizontal')
plt.title("Confusion Matrix — Random Forest")
plt.show()

# --- Важность признаков ---
importances = best_rf.feature_importances_
feature_names = X_train.columns

# Сортируем по убыванию важности
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()