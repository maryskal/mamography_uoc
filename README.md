# Mamography UOC

Este repositorio contiene un proyecto orientado al análisis de imágenes mamográficas utilizando técnicas de aprendizaje profundo. El objetivo es implementar modelos eficientes para clasificar las imágenes en categorías (normal, benigna, maligna) y explorar diferentes enfoques para mejorar el rendimiento mediante ajuste de hiperparámetros, balanceo de clases y técnicas de preprocesamiento.

---

## Estructura del Proyecto

```
mamography_uoc/
├── datasets/                     # Contiene los archivos de datos (TFRecords, imágenes, etiquetas, etc.)
├── preprocess/                   # Scripts para preprocesamiento y manejo de datos
│   ├── dataset_fx.py             # Funciones para manipulación de datasets (TFRecords, concatenación, etc.)
│   ├── load_data.py              # Carga y preparación de datasets (train, val, test)
│   ├── load_models.py            # Construcción de modelos CNN y ViT
│   └── preprocess_fx.py          # Funciones de preprocesamiento de imágenes y etiquetas
├── results/                      # Resultados generados (modelos entrenados, predicciones, métricas)
├── constants.py                  # Definición de constantes globales
├── hyperparameter_tunning.py     # Código para ajuste de hiperparámetros
├── main.py                       # Script principal para entrenamiento de modelos
├── predict.py                    # Script para predicciones con modelos entrenados
├── requirements.txt              # Dependencias necesarias para ejecutar el proyecto
└── .gitignore                    # Archivos y carpetas excluidos del control de versiones
```

---

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/mamography_uoc.git
   cd mamography_uoc
   ```

2. Crea un entorno virtual y activa:
   ```bash
   python -m venv venv
   source venv/bin/activate    # En Linux/Mac
   venv\Scripts\activate       # En Windows
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso

### Entrenamiento

Para entrenar un modelo (por defecto CNN):
```bash
python main.py --model CNN --learning_rate 1e-4 --frozen_proportion 0.8 --loops 3
```

Parámetros:
- `--model`: Modelo a entrenar (`CNN` o `ViT`).
- `--learning_rate`: Tasa de aprendizaje.
- `--frozen_proportion`: Proporción de capas congeladas en el backbone.
- `--loops`: Número de repeticiones del entrenamiento.

### Predicción

Para realizar predicciones con un modelo entrenado:
```bash
python predict.py --model [pesos de tu modelo ya entrenado en .h5]
```

---

## Funcionalidades Principales

1. **Preprocesamiento de Imágenes**:
   - Se aplican técnicas como CLAHE para mejorar el contraste.
   - Normalización y redimensionamiento a 256x256 píxeles.

2. **Modelos**:
   - Implementaciones de CNN con EfficientNet y Vision Transformer (ViT).
   - Uso de estrategias de "fine-tuning" mediante congelación parcial del backbone.

3. **Ajuste de Hiperparámetros**:
   - Optimización de hiperparámetros mediante Mango y búsqueda en espacios definidos.

4. **Manejo de Datos**:
   - Balanceo de clases con pesos ajustados.
   - Generación y manipulación de datasets en formato TFRecord.

---

## Resultados

- Los resultados de los entrenamientos y las métricas se almacenan en `./results/`.
- Los pesos de los modelos entrenados se guardan como archivos `.h5`.
- El ajuste de hiperparámetros genera un resumen en `hyperparameter_tuning_results.csv`.

---

## Requisitos

- Python 3.8 o superior
- Librerías especificadas en `requirements.txt` (ej.: TensorFlow, Keras, Scikit-learn, etc.).

---

## Licencia

Este proyecto se distribuye bajo MIT.

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor, crea un fork del repositorio, realiza tus cambios y envía un pull request.

---

## Contacto

Para dudas o comentarios, puedes contactar a [maryskal.projects@gmail.com].
