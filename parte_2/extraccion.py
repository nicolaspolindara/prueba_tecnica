import os
import json
import time
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
 
import cv2
import vertexai
from dotenv import load_dotenv
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Image, GenerationConfig

# ==========================================
# 0. CONFIG
# ==========================================
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
RUTA_CREDENCIALES = os.getenv("RUTA_CREDENCIALES")
MODEL_NAME = os.getenv("MODEL_NAME")
DIR_IMAGENES = os.getenv("DIR_IMAGENES")
DIR_SALIDA = os.getenv("DIR_SALIDA")

MAX_RETRIES = int(os.getenv("MAX_RETRIES"))
SLEEP_BETWEEN_FILES = float(os.getenv("SLEEP_BETWEEN_FILES"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS"))

# Estimación opcional de costo (USD por 1M tokens)
INPUT_PRICE_PER_1M = float(os.getenv("INPUT_PRICE_PER_1M"))
OUTPUT_PRICE_PER_1M = float(os.getenv("OUTPUT_PRICE_PER_1M"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("extraccion")

# Validación fail-fast
for var_name, var_value in {
    "PROJECT_ID": PROJECT_ID,
    "LOCATION": LOCATION,
    "RUTA_CREDENCIALES": RUTA_CREDENCIALES,
}.items():
    if not var_value:
        raise ValueError(f"Falta variable de entorno requerida: {var_name}")

credenciales = service_account.Credentials.from_service_account_file(RUTA_CREDENCIALES)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credenciales)

model = GenerativeModel(MODEL_NAME)

generation_config = GenerationConfig(
    response_mime_type="application/json",
    temperature=0.1,
)

PROMPT_EXTRACCION = """
Eres un sistema experto en extracción de formularios escaneados.

Devuelve EXCLUSIVAMENTE un JSON válido con esta estructura:
{
  "tipo_documento": "string | null",
  "campos": {
    "clave": "valor | null"
  },
  "tablas": [
    {
      "nombre": "string",
      "filas": [
        {"columna_1": "valor", "columna_2": "valor"}
      ]
    }
  ],
  "warnings": ["string"]
}

Reglas:
1) Extrae solo pares clave-valor explícitos (pregunta -> respuesta).
2) NO inventes etiquetas; usa texto literal o casi literal visible.
3) Si la etiqueta está ilegible/vacía, NO generes clave (agrega warning).
4) Si una clave se repite en distintas secciones, desambigua con contexto breve.
5) Unifica respuestas fragmentadas en un único valor.
6) No incluyas encabezados sueltos sin valor.
7) Si hay duda fuerte o conflicto de lectura, regístralo en warnings.
8) No agregues texto fuera del JSON.
"""

def estimar_costo(input_tokens: int, output_tokens: int) -> float:
    INPUT_PRICE = 1.5
    OUTPUT_PRICE = 9
    return (input_tokens / 1_000_000) * INPUT_PRICE + (output_tokens / 1_000_000) * OUTPUT_PRICE

def _extraer_json_de_texto(texto: str) -> Dict[str, Any]:
    if not texto:
        raise ValueError("Respuesta vacía del modelo")

    limpio = texto.strip()

    # Remover fences ```json ... ```
    limpio = re.sub(r"^```(?:json)?\s*", "", limpio, flags=re.IGNORECASE).strip()
    limpio = re.sub(r"\s*```$", "", limpio).strip()

    # Intento directo
    try:
        return json.loads(limpio)
    except Exception:
        pass

    # Buscar primer bloque { ... }
    ini = limpio.find("{")
    fin = limpio.rfind("}")
    if ini != -1 and fin != -1 and fin > ini:
        candidato = limpio[ini:fin + 1]
        return json.loads(candidato)

    raise ValueError("No se pudo parsear JSON de la respuesta")

def _cargar_imagen_o_pdf(ruta: str):
    path = Path(ruta)
    suf = path.suffix.lower()

    if suf in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"No se pudo cargar imagen: {ruta}")
        return img

    if suf == ".pdf":
        if not PDF2IMAGE_AVAILABLE:
            raise ValueError(
                "PDF detectado pero falta dependencia pdf2image. "
                "Instala: pip install pdf2image (y Poppler en Windows)."
            )
        paginas = convert_from_path(str(path), dpi=220, first_page=1, last_page=1)
        if not paginas:
            raise ValueError(f"No se pudo renderizar PDF: {ruta}")
        pil_img = paginas[0].convert("RGB")
        img = cv2.cvtColor(
            __import__("numpy").array(pil_img),
            cv2.COLOR_RGB2BGR
        )
        return img

    raise ValueError(f"Formato no soportado: {suf}")

def mejorar_imagen(ruta: str) -> bytes:
    img = _cargar_imagen_o_pdf(ruta)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=20)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(denoised)
        final = cv2.adaptiveThreshold(
            contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
    except Exception:
        logger.warning(f"Preprocesamiento avanzado falló en {ruta}. Aplicando fallback.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    ok, buffer = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Error al codificar imagen preprocesada")
    return buffer.tobytes()

def _normalizar_clave(k: Any) -> str:
    k = "" if k is None else str(k).strip()
    k = re.sub(r"\s+", " ", k)
    return k

def _firma_clave(k: str) -> str:
    k = k.lower().strip()
    k = re.sub(r"\s*\([^)]*\)\s*", " ", k)
    k = re.sub(r"__\d+$", "", k)
    k = re.sub(r"[^\w\s]", "", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k

def _es_clave_ruidosa(k: str) -> bool:
    if len(k) < 3:
        return True
    if re.fullmatch(r"[\W_]+", k or ""):
        return True
    return False

def _merge_valores(v1: Any, v2: Any) -> Any:
    s1 = "" if v1 is None else str(v1).strip()
    s2 = "" if v2 is None else str(v2).strip()
    if not s1:
        return s2 or None
    if not s2:
        return s1 or None
    if s2.lower() in s1.lower():
        return s1
    if s1.lower() in s2.lower():
        return s2
    return f"{s1} | {s2}"

def _normalizar_salida_modelo(datos: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(datos, dict):
        raise ValueError("La salida no es un objeto JSON")

    tipo_documento = datos.get("tipo_documento")
    if tipo_documento is not None:
        tipo_documento = str(tipo_documento).strip() or None

    campos_limpios: Dict[str, Any] = {}
    firmas_usadas: Dict[str, str] = {}
    campos = datos.get("campos", {})
    if isinstance(campos, dict):
        for k, v in campos.items():
            key = _normalizar_clave(k)
            if _es_clave_ruidosa(key):
                continue

            if isinstance(v, list):
                vv = " | ".join(str(x).strip() for x in v if str(x).strip()) or None
            elif v is None:
                vv = None
            else:
                vv = str(v).strip() or None

            firma = _firma_clave(key)
            if firma in firmas_usadas:
                k_existente = firmas_usadas[firma]
                campos_limpios[k_existente] = _merge_valores(campos_limpios.get(k_existente), vv)
            else:
                campos_limpios[key] = vv
                firmas_usadas[firma] = key

    tablas_limpias: List[Dict[str, Any]] = []
    tablas = datos.get("tablas", [])
    if isinstance(tablas, list):
        for t in tablas:
            if not isinstance(t, dict):
                continue
            nombre = str(t.get("nombre", "tabla")).strip() or "tabla"
            filas_limpias = []
            for fila in t.get("filas", []):
                if not isinstance(fila, dict):
                    continue
                fila_limpia = {str(c).strip(): (None if v is None else str(v).strip()) for c, v in fila.items() if str(c).strip()}
                if fila_limpia:
                    filas_limpias.append(fila_limpia)
            tablas_limpias.append({"nombre": nombre, "filas": filas_limpias})

    warnings_limpios = []
    warnings = datos.get("warnings", [])
    if isinstance(warnings, list):
        warnings_limpios = [str(w).strip() for w in warnings if str(w).strip()]

    return {
        "tipo_documento": tipo_documento,
        "campos": campos_limpios,
        "tablas": tablas_limpias,
        "warnings": warnings_limpios,
    }

def procesar_formulario(ruta_imagen: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nombre = os.path.basename(ruta_imagen)
    logger.info(f"Procesando: {nombre}")

    inicio = time.time()
    metricas = {
        "archivo": nombre,
        "status": "ok",
        "tiempo_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "costo_estimado_usd": 0.0,
        "error": None,
    }

    try:
        imagen_bytes = mejorar_imagen(ruta_imagen)
        imagen = Image.from_bytes(imagen_bytes)
    except Exception as e:
        metricas["status"] = "error_preprocesamiento"
        metricas["error"] = str(e)
        metricas["tiempo_s"] = round(time.time() - inicio, 3)
        return {"error": f"Error de imagen: {e}"}, metricas

    ultimo_error = None
    for intento in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [imagen, PROMPT_EXTRACCION],
                generation_config=generation_config
            )

            texto_respuesta = getattr(response, "text", "") or ""
            datos_raw = _extraer_json_de_texto(texto_respuesta)
            datos = _normalizar_salida_modelo(datos_raw)

            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            metricas["input_tokens"] = int(input_tokens)
            metricas["output_tokens"] = int(output_tokens)
            metricas["costo_estimado_usd"] = round(estimar_costo(input_tokens, output_tokens), 8)
            metricas["tiempo_s"] = round(time.time() - inicio, 3)
            return datos, metricas
        except Exception as e:
            ultimo_error = str(e)
            logger.warning(f"{nombre} | intento {intento}/{MAX_RETRIES} falló: {ultimo_error}")
            if intento < MAX_RETRIES:
                time.sleep(min(8.0, 1.5 ** intento))

    metricas["status"] = "error_modelo"
    metricas["error"] = ultimo_error
    metricas["tiempo_s"] = round(time.time() - inicio, 3)
    return {"error": f"Fallo tras reintentos: {ultimo_error}"}, metricas

def procesar_carpeta(directorio_imagenes: str, directorio_salida: str):
    if not os.path.isdir(directorio_imagenes):
        raise ValueError(f"Directorio de entrada inválido: {directorio_imagenes}")
    os.makedirs(directorio_salida, exist_ok=True)

    ruta_reporte = os.path.join(directorio_salida, "_reporte_ejecucion.jsonl")
    ruta_resumen = os.path.join(directorio_salida, "_resumen_ejecucion.json")
    extensiones = (".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tif", ".tiff")
    archivos = [f for f in os.listdir(directorio_imagenes) if f.lower().endswith(extensiones)]
    archivos.sort()

    total = 0
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    total_time = 0.0

    resultados: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    def _procesar_un_archivo(archivo: str):
        ruta = os.path.join(directorio_imagenes, archivo)
        datos_json, metricas = procesar_formulario(ruta)
        return archivo, datos_json, metricas

    if MAX_WORKERS > 1 and len(archivos) > 1:
        logger.info(f"Procesamiento paralelo activado (MAX_WORKERS={MAX_WORKERS})")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_procesar_un_archivo, a): a for a in archivos}
            for f in as_completed(futures):
                archivo = futures[f]
                try:
                    resultados.append(f.result())
                except Exception as e:
                    resultados.append((
                        archivo,
                        {"error": f"Error inesperado: {e}"},
                        {
                            "archivo": archivo, "status": "error_inesperado", "tiempo_s": 0.0,
                            "input_tokens": 0, "output_tokens": 0, "costo_estimado_usd": 0.0, "error": str(e)
                        }
                    ))
    else:
        for archivo in archivos:
            resultados.append(_procesar_un_archivo(archivo))
            time.sleep(SLEEP_BETWEEN_FILES)

    resultados.sort(key=lambda x: x[0])

    with open(ruta_reporte, "w", encoding="utf-8") as rep:
        for archivo, datos_json, metricas in resultados:
            salida = os.path.join(directorio_salida, f"{Path(archivo).stem}.json")
            with open(salida, "w", encoding="utf-8") as f:
                json.dump(datos_json, f, indent=2, ensure_ascii=False)
            rep.write(json.dumps(metricas, ensure_ascii=False) + "\n")

            total += 1
            total_tokens_in += metricas["input_tokens"]
            total_tokens_out += metricas["output_tokens"]
            total_cost += metricas["costo_estimado_usd"]
            total_time += metricas["tiempo_s"]

            logger.info(
                f" -> {archivo} | {metricas['status']} | "
                f"{metricas['tiempo_s']:.2f}s | in={metricas['input_tokens']} out={metricas['output_tokens']}"
            )
            time.sleep(SLEEP_BETWEEN_FILES)

    resumen = {
        "documentos": total,
        "tiempo_total_s": round(total_time, 3),
        "tiempo_promedio_s": round((total_time / total), 3) if total else 0.0,
        "tokens_entrada": int(total_tokens_in),
        "tokens_salida": int(total_tokens_out),
        "costo_estimado_total_usd": round(total_cost, 8),
    }
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    if total > 0:
        logger.info("=== RESUMEN ===")
        logger.info(f"Docs: {total}")
        logger.info(f"Tiempo total: {total_time:.2f}s | Promedio: {total_time/total:.2f}s")
        logger.info(f"Tokens entrada/salida: {total_tokens_in}/{total_tokens_out}")
        logger.info(f"Costo estimado total (USD): {total_cost:.6f}")
        logger.info(f"Resumen guardado en: {ruta_resumen}")

if __name__ == "__main__":
    procesar_carpeta(DIR_IMAGENES, DIR_SALIDA)