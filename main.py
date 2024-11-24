import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Clasificación de frutas con k-NN
def classify_fruit():
    data = np.array([
        [150, 7, 1],  # Manzana
        [170, 8, 1],  # Manzana
        [130, 6, 0],  # Naranja
        [180, 8.5, 0] # Naranja
    ])

    X = data[:, :2]  # Características (Peso, Diámetro)
    y = data[:, 2]   # Etiquetas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    new_fruit = np.array([[160, 7.5]])  # Nueva fruta
    prediction = knn.predict(new_fruit)
    fruit_type = "Manzana" if prediction[0] == 1 else "Naranja"

    print(f"Resultado k-NN: La nueva fruta probablemente es una: {fruit_type}")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label="Datos conocidos")
    plt.scatter(new_fruit[0, 0], new_fruit[0, 1], color='red', label="Fruta nueva", s=100)
    plt.xlabel("Peso (g)")
    plt.ylabel("Diámetro (cm)")
    plt.legend()
    plt.title("Clasificación de Frutas con k-NN")
    plt.show()

# 2. Clasificación de correos con Regresión Logística
def classify_email():
    data = np.array([
        [10, 1, 1],  # Spam
        [2, 0, 0],   # No Spam
        [15, 1, 1],  # Spam
        [1, 0, 0]    # No Spam
    ])

    X = data[:, :2]  # Características
    y = data[:, 2]   # Etiquetas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Resultado Regresión Logística: Precisión del modelo: {accuracy * 100:.2f}%")

# 3. Decisión de jugar tenis con Árboles de Decisión y Random Forest
def play_tennis_decision():
    data = pd.DataFrame({
        "Clima": ["Soleado", "Nublado", "Lluvioso", "Soleado"],
        "Temperatura": ["Alta", "Alta", "Baja", "Baja"],
        "Humedad": ["Alta", "Alta", "Baja", "Alta"],
        "JugarTenis": [0, 1, 1, 0]
    })

    data_encoded = pd.get_dummies(data.drop(columns=["JugarTenis"]))
    y = data["JugarTenis"]

    X_train, X_test, y_train, y_test = train_test_split(data_encoded, y, test_size=0.25, random_state=42)

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plot_tree(tree_clf, feature_names=data_encoded.columns, class_names=["No", "Sí"], filled=True)
    plt.title("Árbol de Decisión - Jugar Tenis")
    plt.show()

    y_pred = tree_clf.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred)
    print(f"Resultado Árbol de Decisión: Precisión del modelo: {accuracy_tree * 100:.2f}%")

    rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred_rf = rf_clf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Resultado Random Forest: Precisión del modelo: {accuracy_rf * 100:.2f}%")

# Ejecución de los ejemplos
if __name__ == "__main__":
    print("1. Clasificación de frutas con k-NN")
    classify_fruit()
    print("\n2. Clasificación de correos con Regresión Logística")
    classify_email()
    print("\n3. Decisión de jugar tenis con Árboles de Decisión y Random Forest")
    play_tennis_decision()
