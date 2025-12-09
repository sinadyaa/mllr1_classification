from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

csv_path = r"D:\!учеба\ML\lr1\heart_disease2.csv"
# Чтение CSV файла
df = pd.read_csv(csv_path)

X = df.drop('HadDiabetes', axis=1)  # Все кроме целевой переменной
y = df['HadDiabetes']               # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify для баланса классов
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"y_train: {y_train.shape}, y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}, y_test: {y_test.shape}")

# Сохраняем тестовые без изменений
X_test.to_csv(r"D:\!учеба\ML\lr1\heart_disease_xtest.csv", index=False, encoding='utf-8-sig')
y_test.to_csv(r"D:\!учеба\ML\lr1\heart_disease_ytest.csv", index=False, encoding='utf-8-sig')

def s_smote(X, y, k=5):

    X_np, y_np = np.array(X), np.array(y)
    X_resampled, y_resampled = X_np.copy(), y_np.copy()

    # Считаем количество образцов в каждом классе
    class_counts = np.bincount(y_np)
    target_count = int(np.max(class_counts))  # Целевое количество - как у мажоритарного класса

    # Балансируем каждый миноритарный класс
    for class_label in np.unique(y_np):
        X_class = X_np[y_np == class_label]
        n_samples = target_count - len(X_class)  # Сколько нужно сгенерировать

        if n_samples <= 0:
            continue  # Пропускаем мажоритарный класс

        # Находим k ближайших соседей
        nn = NearestNeighbors(n_neighbors=min(k, len(X_class)))
        nn.fit(X_class)

        # Генерируем синтетические образцы
        for _ in range(n_samples):
            idx = np.random.randint(len(X_class))
            neighbors = nn.kneighbors([X_class[idx]], return_distance=False)[0]
            neighbor_idx = neighbors[np.random.randint(1, min(k, len(neighbors)))]
            
            # Создаем синтетический образец
            diff = X_class[neighbor_idx] - X_class[idx]
            synthetic = X_class[idx] + np.random.random() * diff

            # Добавляем к результату
            X_resampled = np.vstack([X_resampled, synthetic])
            y_resampled = np.append(y_resampled, class_label)

    return X_resampled, y_resampled

# Применение функции
X_train_bal, y_train_bal = s_smote(X_train, y_train, k=3)

# Проверим результат
print("После балансировки:")
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique, counts)))

print(f"X_train_bal shape: {X_train_bal.shape}")
print(f"y_train_bal shape: {y_train_bal.shape}")

# Сохранение результатов
pd.DataFrame(X_train_bal).to_csv(r"D:\!учеба\ML\lr1\heart_disease_xtrain_bal.csv", index=False, encoding='utf-8-sig')
pd.Series(y_train_bal).to_csv(r"D:\!учеба\ML\lr1\heart_disease_ytrain_bal.csv", index=False, encoding='utf-8-sig')



print("✅ Все выборки успешно сохранены!")