# Ejemplos de Machine Learning con Python

Este proyecto incluye tres ejemplos básicos de Machine Learning:

1. **Clasificación de frutas** usando k-Nearest Neighbors (k-NN).
2. **Clasificación de correos** como spam o no spam usando Regresión Logística.
3. **Decisión de jugar tenis** usando Árboles de Decisión y Random Forest.

## Requisitos Previos

- Python 3.8 o superior.
- Bibliotecas necesarias: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Instalación

Sigue los pasos a continuación para configurar y ejecutar el proyecto:

1. Clona este repositorio o descarga el archivo ZIP:
   ```bash
   git clone <url_del_repositorio>
   cd <nombre_del_repositorio>
   ```

2. Instala las dependencias usando pip:
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```

## Uso

Ejecuta el archivo principal para probar los tres ejemplos:
  ```bash
  python main.py
  ```

## EJERCICIOS
### Ejercicio 1: Clasificación de Frutas (k-NN)

    Descripción: Clasifica una fruta como manzana o naranja basándose en su peso y diámetro.
    Modelo usado: k-Nearest Neighbors (k-NN).
    Resultados: Los resultados se muestran en la consola y se genera un gráfico para visualizar los datos.

### Ejercicio 2: Clasificación de Correos (Regresión Logística)

    Descripción: Predice si un correo es spam o no usando características como el número de palabras en mayúsculas y la presencia de ciertas palabras clave.
    Modelo usado: Regresión Logística.
    Resultados: La precisión del modelo se imprime en la consola.

### Ejercicio 3: Decisión de Jugar Tenis (Árboles de Decisión y Random Forest)

    Descripción: Usa información del clima, temperatura y humedad para decidir si jugar tenis.
    Modelos usados:
    - Árbol de Decisión.
    - Random Forest.
    Resultados: Se genera un gráfico del árbol de decisión y la precisión de ambos modelos se imprime en la consola.