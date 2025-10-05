import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Crear el conjunto de datos
data = {
    "Texto": [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    "Etiqueta": [
        "real", "fake", "real", "fake", "real",
        "fake", "real", "fake", "real", "fake"
    ]
}

# Nuevas noticias a clasificar
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

# Convertir a DataFrame
df = pd.DataFrame(data)

# Vectorización de texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Texto"])
y = df["Etiqueta"]

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluar sobre el conjunto de prueba
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test, labels=["real", "fake"])


# Clasificar nuevas noticias
X_new = vectorizer.transform(nuevas_noticias)
predicciones_nuevas = model.predict(X_new)

# Crear informe
informe = f"""
Precisión total (accuracy): {accuracy:.2f}

Matriz de confusión:
Real | Fake
{conf_matrix[0][0]}    | {conf_matrix[0][1]}
{conf_matrix[1][0]}    | {conf_matrix[1][1]}

Clasificación de nuevas noticias:
1. '{nuevas_noticias[0]}' → {predicciones_nuevas[0]}
2. '{nuevas_noticias[1]}' → {predicciones_nuevas[1]}
"""

# Guardar informe
output_path = "/mnt/data/informe_naive_bayes.txt"
os.makedirs("/mnt/data", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(informe)

print(informe)
