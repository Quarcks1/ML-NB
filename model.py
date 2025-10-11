import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ======================
# 1️ Cargar datasets
# ======================
df_train = pd.read_csv('./Data/dataset10000.csv')  # Entrenamiento grande
df_test = pd.read_csv('./Data/dataset1000.csv')    # Test grande

features_columns = ['Director', 'Production', 'User', 'Genre']
target_column = 'Rank'

x_train = df_train[features_columns]
y_train = df_train[target_column]

x_test = df_test[features_columns]
y_test = df_test[target_column]

# ======================
# 2️ Función Hashing Trick
# ======================
def hashing_transform(df, columns, n_features=2**12):
    """
    Convierte columnas categóricas en matriz dispersa usando Hashing Trick
    n_features: número de columnas finales (2^12 = 4096)
    alternate_sign=False asegura valores no negativos para MultinomialNB
    """
    hasher = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False)
    
    # Combinar todas las columnas en strings "col=value"
    combined = df[columns].astype(str).agg(lambda x: [f"{col}={val}" for col, val in zip(columns, x)], axis=1)
    
    X_hashed = hasher.transform(combined)
    return X_hashed

# Transformar datasets
X_train_hashed = hashing_transform(x_train, features_columns, n_features=2**12)
X_test_hashed = hashing_transform(x_test, features_columns, n_features=2**12)

# ======================
# 3️ Entrenamiento MultinomialNB
# ======================
model = MultinomialNB(alpha=0.5)
model.fit(X_train_hashed, y_train)

# ======================
# 4️ Evaluación en test
# ======================
y_pred = model.predict(X_test_hashed)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en test set grande: {accuracy:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# ======================
# 5️ Visualizar distribución de clases
# ======================
def show_class_distribution(y, title="Distribución de clases"):
    y.value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Frecuencia")
    plt.show()

show_class_distribution(y_train, "Distribución de clases en dataset de entrenamiento")
show_class_distribution(y_test, "Distribución de clases en dataset de test")
