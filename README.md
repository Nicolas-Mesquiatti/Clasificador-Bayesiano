# 📰 Clasificador de Noticias: ¿Real o Fake?

Este proyecto implementa un modelo de aprendizaje automático basado en **Naive Bayes** para detectar si una noticia es *real* o *falsa* según su contenido textual.

---

##  Objetivo

Desarrollar un sistema automático que:

- Entrene un modelo Naive Bayes con un conjunto de noticias etiquetadas.
- Evalúe su rendimiento sobre datos de prueba.
- Clasifique nuevas noticias no vistas previamente.

---

##  Estructura del proyecto

- `bayesianno.py`: Código principal que entrena el modelo, evalúa su precisión y clasifica nuevas noticias.
- `informebayesiano.pdf`: Informe generado automáticamente con los resultados del modelo.
- `README.md`: Este archivo.

---

##  Datos de entrenamiento

El conjunto de datos incluye ejemplos como:

| Texto                                                                 | Etiqueta |
|-----------------------------------------------------------------------|----------|
| El presidente anunció una nueva reforma educativa                    | real     |
| Descubren que la vacuna convierte a las personas en robots           | fake     |
| La NASA confirma el hallazgo de agua en Marte                        | real     |
| Científicos afirman que la Tierra es plana                           | fake     |
| ...                                                                   | ...      |

---

##  Resultados del modelo
Precisión sobre conjunto de prueba: 1.00

Matriz de confusión:

Clasificación de nuevas noticias:

“Nuevo estudio demuestra que el café mejora la memoria” → fake

“Expertos afirman que los gatos pueden hablar con humanos” → fake


##  Limitaciones
El conjunto de datos es pequeño y puede inducir sobreajuste.

Algunas noticias reales pueden ser clasificadas como falsas si contienen vocabulario sensacionalista.


   git clone https://github.com/tu_usuario/clasificador-noticias.git
   cd clasificador-noticias
