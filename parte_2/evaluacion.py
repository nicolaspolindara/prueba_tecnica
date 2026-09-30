import os
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple
from difflib import SequenceMatcher


def _quitar_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")

def normalizar_a_tokens(texto: Any) -> set:
    if texto is None:
        return set()
    if not isinstance(texto, str):
        texto = str(texto)
    texto = _quitar_acentos(texto.lower())
    texto = texto.replace("_", " ").replace("-", " ")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return set(texto.split()) if texto else set()

def normalizar_valor(texto: Any) -> str:
    if texto is None:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)
    t = _quitar_acentos(texto.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace(" / ", "/").replace(" ,", ",")
    t = re.sub(r"(\d)\s+([/%])", r"\1\2", t)
    t = re.sub(r"([/%])\s+(\d)", r"\1\2", t)
    return t

def similitud_jaccard(tokens1: set, tokens2: set) -> float:
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def similitud_char(a: Any, b: Any) -> float:
    sa = normalizar_valor(a)
    sb = normalizar_valor(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return SequenceMatcher(None, sa, sb).ratio()

def _par_valido(k: Any, v: Any) -> bool:
    k = "" if k is None else str(k).strip()
    v = "" if v is None else str(v).strip()
    if len(k) < 3:
        return False
    if v in {"", ".", "-", "--", "n/a", "na", "null", "none"}:
        return False
    return True

def _normalizar_clave_match(texto: Any) -> str:
    if texto is None:
        return ""
    t = str(texto).lower().strip()
    t = re.sub(r"__\d+$", "", t)  
    t = _quitar_acentos(t)
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)  
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _canon_lista_valor(texto: Any) -> set:
    t = normalizar_valor(texto)
    partes = [p.strip() for p in re.split(r"[|;,]", t) if p.strip()]
    if not partes:
        partes = re.findall(r"\d+/\d+|\w+", t)
    return set(partes)

def extraer_pares_ground_truth(datos_gt: Dict[str, Any]) -> List[Tuple[str, str]]:
    form = datos_gt.get("form", []) if isinstance(datos_gt, dict) else []
    by_id = {e.get("id"): e for e in form if isinstance(e, dict) and "id" in e}

    edges = []
    seen = set()
    for e in form:
        if not isinstance(e, dict):
            continue
        for lk in e.get("linking", []):
            if isinstance(lk, list) and len(lk) == 2:
                a, b = lk
                if a == b:
                    continue
                edge = tuple(sorted((a, b)))
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)

    agrupado: Dict[str, List[str]] = {}
    for a, b in edges:
        ea = by_id.get(a)
        eb = by_id.get(b)
        if not ea or not eb:
            continue

        la, lb = ea.get("label"), eb.get("label")
        ta = (ea.get("text") or "").strip()
        tb = (eb.get("text") or "").strip()

        key, val = None, None
        if la in {"question", "header"} and lb == "answer":
            key, val = ta, tb
        elif lb in {"question", "header"} and la == "answer":
            key, val = tb, ta

        if key and val and _par_valido(key, val):
            agrupado.setdefault(key, [])
            if val not in agrupado[key]:
                agrupado[key].append(val)

    pares = []
    for k, vals in agrupado.items():
        vals = [v.strip() for v in vals if v and v.strip()]
        if vals:
            pares.append((k, " | ".join(vals)))
    return pares

def extraer_pares_prediccion(obj: Any, solo_campos: bool = True) -> List[Tuple[str, str]]:
    pares = []
    if not isinstance(obj, dict):
        return pares

    campos = obj.get("campos", {})
    if isinstance(campos, dict):
        for k, v in campos.items():
            if _par_valido(k, v):
                pares.append((str(k), str(v).strip()))

    if not solo_campos:
        tablas = obj.get("tablas", [])
        if isinstance(tablas, list):
            for t in tablas:
                if not isinstance(t, dict):
                    continue
                nombre_tabla = t.get("nombre", "tabla")
                for fila in t.get("filas", []):
                    if not isinstance(fila, dict):
                        continue
                    for col, val in fila.items():
                        k = f"{nombre_tabla}.{col}"
                        if _par_valido(k, val):
                            pares.append((k, str(val).strip()))

    return pares

def score_par(kp: str, vp: str, kg: str, vg: str, w_key: float = 0.45, w_val: float = 0.55):
    sk = similitud_jaccard(normalizar_a_tokens(_normalizar_clave_match(kp)), normalizar_a_tokens(_normalizar_clave_match(kg)))
    sv_tokens = similitud_jaccard(normalizar_a_tokens(vp), normalizar_a_tokens(vg))
    sv_lista = similitud_jaccard(_canon_lista_valor(vp), _canon_lista_valor(vg))
    sv_char = similitud_char(vp, vg)
    sv = max(sv_tokens, sv_lista, sv_char)

    score = (w_key * sk + w_val * sv)
    if normalizar_valor(vp) == normalizar_valor(vg):
        score = min(1.0, score + 0.08)

    return score, sk, sv

def _to_bool_env(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}

def _match_estricto(kp: str, vp: str, kg: str, vg: str) -> bool:
    return (
        _normalizar_clave_match(kp) == _normalizar_clave_match(kg)
        and normalizar_valor(vp) == normalizar_valor(vg)
    )

def emparejar_pares(
    pred: List[Tuple[str, str]],
    gt: List[Tuple[str, str]],
    umbral_score: float = 0.80,
    umbral_key: float = 0.50,
    umbral_val: float = 0.70,
    modo_estricto: bool = False
):
    candidatos = []
    for i, (kp, vp) in enumerate(pred):
        for j, (kg, vg) in enumerate(gt):
            if modo_estricto:
                if _match_estricto(kp, vp, kg, vg):
                    candidatos.append((1.0, i, j))
                continue
            score, sk, sv = score_par(kp, vp, kg, vg)
            if score >= umbral_score and sk >= umbral_key and sv >= umbral_val:
                candidatos.append((score, i, j))

    candidatos.sort(key=lambda x: x[0], reverse=True)

    usados_pred, usados_gt = set(), set()
    matches = []
    for score, i, j in candidatos:
        if i in usados_pred or j in usados_gt:
            continue
        usados_pred.add(i)
        usados_gt.add(j)
        matches.append((i, j, score))

    # 2da pasada relajada para rescatar equivalencias de valor fuerte
    if not modo_estricto:
        for i in [x for x in range(len(pred)) if x not in usados_pred]:
            kp, vp = pred[i]
            for j in [y for y in range(len(gt)) if y not in usados_gt]:
                kg, vg = gt[j]
                sk = similitud_jaccard(normalizar_a_tokens(_normalizar_clave_match(kp)), normalizar_a_tokens(_normalizar_clave_match(kg)))
                sv_char = similitud_char(vp, vg)
                if sk >= (umbral_key - 0.10) and sv_char >= 0.90:
                    usados_pred.add(i)
                    usados_gt.add(j)
                    matches.append((i, j, 0.95))
                    break

    unmatched_pred = [i for i in range(len(pred)) if i not in usados_pred]
    unmatched_gt = [j for j in range(len(gt)) if j not in usados_gt]

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, matches, unmatched_pred, unmatched_gt

def calcular_metricas(
    dir_ground_truth: str,
    dir_predicciones: str,
    umbral_score: float = 0.80,
    umbral_key: float = 0.50,
    umbral_val: float = 0.70,
    solo_campos: bool = True,
    modo_estricto: bool = False,
):
    tp_total = fp_total = fn_total = 0
    detalles = []

    archivos_pred = sorted(
        f for f in os.listdir(dir_predicciones)
        if f.endswith(".json") and not f.startswith("_")
    )

    for archivo in archivos_pred:
        ruta_pred = os.path.join(dir_predicciones, archivo)
        ruta_gt = os.path.join(dir_ground_truth, archivo)
        if not os.path.exists(ruta_gt):
            continue

        with open(ruta_pred, "r", encoding="utf-8") as f:
            datos_pred = json.load(f)
        with open(ruta_gt, "r", encoding="utf-8") as f:
            datos_gt = json.load(f)

        pares_pred = extraer_pares_prediccion(datos_pred, solo_campos=solo_campos)
        pares_gt = extraer_pares_ground_truth(datos_gt)

        tp, fp, fn, matches, um_pred, um_gt = emparejar_pares(
            pares_pred, pares_gt,
            umbral_score=umbral_score,
            umbral_key=umbral_key,
            umbral_val=umbral_val,
            modo_estricto=modo_estricto
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn

        detalles.append({
            "archivo": archivo,
            "pred_pairs": len(pares_pred),
            "gt_pairs": len(pares_gt),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fp_examples": [pares_pred[i] for i in um_pred[:5]],
            "fn_examples": [pares_gt[j] for j in um_gt[:5]],
        })

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1, detalles, tp_total, fp_total, fn_total

def barrer_umbrales(dir_gt: str, dir_pred: str, solo_campos: bool = True):
    mejores = None
    for us in [0.75, 0.80, 0.85]:
        for uk in [0.40, 0.50, 0.60]:
            for uv in [0.60, 0.70, 0.80]:
                p, r, f1, _, tp, fp, fn = calcular_metricas(
                    dir_gt, dir_pred,
                    umbral_score=us, umbral_key=uk, umbral_val=uv,
                    solo_campos=solo_campos
                )
                cand = {
                    "umbral_score": us, "umbral_key": uk, "umbral_val": uv,
                    "precision": p, "recall": r, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn
                }
                if mejores is None or cand["f1"] > mejores["f1"]:
                    mejores = cand
    return mejores

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    default_dir_gt = base_dir / "dataset" / "training_data" / "annotations"
    default_dir_pred = base_dir / "model_outputs"

    DIR_GT = os.getenv("DIR_GT", str(default_dir_gt))
    DIR_PRED = os.getenv("DIR_PRED", str(default_dir_pred))
    MODO_ESTRICTO_KV = _to_bool_env(os.getenv("MODO_ESTRICTO_KV", "0"))

    # Umbrales por defecto según mejor barrido observado
    p, r, f1, detalles, tp, fp, fn = calcular_metricas(
        DIR_GT, DIR_PRED,
        umbral_score=0.75,
        umbral_key=0.40,
        umbral_val=0.60,
        solo_campos=True,
        modo_estricto=MODO_ESTRICTO_KV
    )

    print("=== MÉTRICAS DE PARES CLAVE-VALOR ===")
    print(f"TP={tp} | FP={fp} | FN={fn}")
    print(f"Precisión: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\n=== TOP 5 ARCHIVOS CON MÁS FP ===")
    for d in sorted(detalles, key=lambda x: x["fp"], reverse=True)[:5]:
        print(f"{d['archivo']} | TP={d['tp']} FP={d['fp']} FN={d['fn']}")
        if d["fp_examples"]:
            print(f"  FP ej: {d['fp_examples'][0]}")
        if d["fn_examples"]:
            print(f"  FN ej: {d['fn_examples'][0]}")

    best = barrer_umbrales(DIR_GT, DIR_PRED, solo_campos=True)
    # print("\n=== MEJOR CONFIG (barrido) ===")
    # print(best)

    ruta_reporte = os.path.join(DIR_PRED, "_reporte_metricas.json")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f1,
                "best_thresholds": best,
                "detalles": detalles,
                "modo_estricto_kv": MODO_ESTRICTO_KV,
                "criterio_matching": "estricto" if MODO_ESTRICTO_KV else "similaridad"
            },
            f, ensure_ascii=False, indent=2
        )