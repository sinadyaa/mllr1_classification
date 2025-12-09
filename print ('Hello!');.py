import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

csv_path = r"D:\!учеба\ML\lr1\heart_disease.csv"
# Чтение CSV файла
df = pd.read_csv(csv_path)

# print(f"Общий размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")

# Удаление дубликатов
df_cleaned = df.drop_duplicates()
# print(f"После удаления дубликатов: {df_cleaned.shape}")

# Удаление строк с пропусками
df_final = df_cleaned.dropna().copy()
# print(f"После удаления пропусков: {df_final.shape}")

# =============================================================================
# СОКРАЩЕНИЕ ВЫБОРКИ ДО 40 ТЫСЯЧ С СОХРАНЕНИЕМ ПРОПОРЦИЙ
# =============================================================================

print(f"Исходный размер данных: {df_final.shape}")

# Если данных больше 40,000, сокращаем выборку
if len(df_final) > 40000:
    # Используем train_test_split для сокращения с сохранением пропорций
    df_reduced, _ = train_test_split(
        df_final, 
        train_size=40000, 
        random_state=42, 
        stratify=df_final['HadDiabetes']  # сохраняем пропорции целевой переменной
    )
    df_final = df_reduced

print(f"Сокращенный размер данных: {df_final.shape}")
print("Распределение HadDiabetes после сокращения:")
print(df_final['HadDiabetes'].value_counts(normalize=True) * 100)

# =============================================================================
# ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================

# Разделение колонок по типам
numerical_cols = df_final.select_dtypes(include=['float64']).columns.tolist()
categorical_cols = df_final.select_dtypes(include=['object']).columns.tolist()

# Кодирование категориальных данных
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders[col] = le

# StandardScaler для нормальных распределений
standard_cols = ['BMI', 'SleepHours']
if all(col in df_final.columns for col in standard_cols):
    scaler_standard = StandardScaler()
    df_final[standard_cols] = scaler_standard.fit_transform(df_final[standard_cols])

# 2. RobustScaler для данных с выбросами
robust_cols = ['PhysicalHealthDays', 'MentalHealthDays']
if all(col in df_final.columns for col in robust_cols):
    scaler_robust = RobustScaler()
    df_final[robust_cols] = scaler_robust.fit_transform(df_final[robust_cols])

# =============================================================================
# СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ В ФАЙЛ
# =============================================================================

# Сохраняем обработанный датафрейм
output_path = r"D:\!учеба\ML\lr1\heart_disease2.csv"
df_final.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ Обработанные данные сохранены в: {output_path}")
print(f"📊 Размер сохраненного файла: {df_final.shape}")


# Настройка стиля
sns.set(style="whitegrid")
#графики колонки
# Создаем окно с тремя подграфиками
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# GeneralHealth vs HadHeartAttack
sns.barplot(
    x='GeneralHealth', y='HadHeartAttack',
    data=df_final, estimator='mean', ci=None, ax=axes[0]
)
axes[0].set_title('GeneralHealth vs HadHeartAttack')
axes[0].set_xlabel('General Health')
axes[0].set_ylabel('Had Heart Attack')

# 2HadDiabetes vs HadHeartAttack
sns.barplot(
    x='HadDiabetes', y='HadHeartAttack',
    data=df_final, estimator='mean', ci=None, ax=axes[1]
)
axes[1].set_title('HadDiabetes vs HadHeartAttack')
axes[1].set_xlabel('Had Diabetes')
axes[1].set_ylabel('Had Heart Attack')

# 3 SleepHours vs HadHeartAttack
sns.barplot(
    x='HadHeartAttack', y='SleepHours',
    data=df_final, estimator='mean', ci=None, ax=axes[2]
)
axes[2].set_title('SleepHours vs HadHeartAttack')
axes[2].set_xlabel('Had Heart Attack')
axes[2].set_ylabel('Sleep Hours')

# Уплотняем расположение и показываем
plt.tight_layout()
plt.show()

#диаграммы рассеивания
# Создание фигуры с 3 подграфиками в 1 ряду
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 1. SleepHours vs MentalHealthDays ---
sns.scatterplot(
    data=df_final, 
    x='SleepHours', 
    y='MentalHealthDays', 
    alpha=0.6, 
    ax=axes[0]
)
axes[0].set_title('SleepHours vs MentalHealthDays')
axes[0].set_xlabel('SleepHours')
axes[0].set_ylabel('MentalHealthDays')

# --- 2. SleepHours vs PhysicalHealthDays ---
sns.scatterplot(
    data=df_final, 
    x='SleepHours', 
    y='PhysicalHealthDays', 
    alpha=0.6, 
    ax=axes[1]
)
axes[1].set_title('SleepHours vs PhysicalHealthDays')
axes[1].set_xlabel('SleepHours')
axes[1].set_ylabel('PhysicalHealthDays')

# --- 3. PhysicalHealthDays vs BMI ---
sns.scatterplot(
    data=df_final, 
    x='PhysicalHealthDays', 
    y='BMI', 
    alpha=0.6, 
    ax=axes[2]
)
axes[2].set_title('PhysicalHealthDays vs BMI')
axes[2].set_xlabel('PhysicalHealthDays')
axes[2].set_ylabel('BMI')


# Компоновка
plt.tight_layout()
plt.show()

# === 2. Ящики с усами (Boxplots) ===
columns = ['BMI', 'SleepHours', 'PhysicalHealthDays', 'MentalHealthDays']

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, col in enumerate(columns):
    sns.boxplot(x=df_final[col], ax=axes[i])
    axes[i].set_title(f'{col}', fontsize=11)
    axes[i].set_xlabel('Значения')

plt.tight_layout()
plt.show()


# === 3. Гистограммы (Histograms) ===
df_cleaned[numerical_cols].hist(figsize=(10, 6), bins=20, edgecolor='black')
plt.suptitle('Гистограммы распределения числовых признаков')
plt.tight_layout()
plt.show()

# === 4. Матрица корреляций ===
plt.figure(figsize=(10, 8))
corr_matrix = df_final.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица корреляций всех признаков')
plt.tight_layout()
plt.show()
