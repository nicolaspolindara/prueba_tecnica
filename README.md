# Proyecto de extracción y evaluación de formularios (IA generativa)

Este repositorio implementa un pipeline para extraer información estructurada desde formularios escaneados y evaluarla contra anotaciones humanas en formato clave-valor.

## Arquitectura

El flujo está dividido en dos módulos principales:

1. `parte_2/extraccion.py`
   - Carga imágenes/PDF.
   - Preprocesa documentos (denoise, contraste, binarización; con fallback OTSU).
   - Invoca un modelo multimodal de Vertex AI con prompt estructurado.
   - Normaliza salida JSON (`tipo_documento`, `campos`, `tablas`, `warnings`).
   - Soporta paralelismo con `ThreadPoolExecutor`.
   - Guarda:
     - JSON por documento en `model_outputs/`
     - `_reporte_ejecucion.jsonl` (detalle por archivo)
     - `_resumen_ejecucion.json` (resumen agregado)

2. `parte_2/evaluacion.py`
   - Construye ground truth a partir de `annotations/`.
   - Extrae pares predichos desde `campos` (y opcionalmente `tablas`).
   - Calcula matching entre pares clave-valor:
     - modo por similitud semantica
     - modo estricto - por pares
   - Reporta `TP`, `FP`, `FN`, precisión, recall y F1.
   - Genera `_reporte_metricas.json`.

## Requisitos

- Python 3.10+
- API KEY habilitada para conexion a modelos de IA
- Dependencias Python:
  - `opencv-python`
  - `python-dotenv`
  - `google-auth`
  - `google-cloud-aiplatform`
  - `vertexai`
  - `pdf2image` 

## Instalación

1. Crear y activar entorno virtual.
2. Instalar dependencias:

```bash
pip install opencv-python python-dotenv google-auth google-cloud-aiplatform vertexai pdf2image
```

3. Crear archivo `.env` en la raíz del proyecto.

## Variables de entorno

Configurar en `.env` (ejemplo):

```env
# Vertex AI
PROJECT_ID=tu-project-id
LOCATION=us-central1
RUTA_CREDENCIALES=C:/ruta/segura/service-account.json
MODEL_NAME=gemini-3.5-flash

# I/O
DIR_IMAGENES=C:/ruta/proyecto/dataset/testing_data/images
DIR_SALIDA=C:/ruta/proyecto/model_outputs

# Ejecución
MAX_RETRIES=3
SLEEP_BETWEEN_FILES=0.5
MAX_WORKERS=3

## Ejecución

### 1) Extracción

```bash
python parte_2/extraccion.py
```

Salida:
- JSON por documento en `model_outputs/`
- `model_outputs/_reporte_ejecucion.jsonl`
- `model_outputs/_resumen_ejecucion.json`

### 2) Evaluación

```bash
python parte_2/evaluacion.py
```

Salida:
- métricas en consola (`TP`, `FP`, `FN`, precisión, recall, F1)
- `model_outputs/_reporte_metricas.json`

## Métricas

- `TP`: pares clave-valor correctos recuperados.
- `FP`: pares predichos que no corresponden al ground truth.
- `FN`: pares del ground truth no recuperados.

Fórmulas:
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * precision * recall / (precision + recall)`
