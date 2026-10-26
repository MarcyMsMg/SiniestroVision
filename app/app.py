"""
SiniestroVision — evaluación de daños vehiculares para aseguradoras.

Flujo:  foto -> modelo TFLite (severidad) -> estimación de costo -> decisión
        -> caso guardado (foto en S3, caso en DynamoDB) -> revisión del liquidador.

Variables de entorno (todas opcionales; sin ellas todo funciona en memoria, útil para pruebas):
  BUCKET_S3        bucket donde se guardan las fotos (privadas; se ven con enlaces temporales)
  TABLA_DYNAMO     tabla DynamoDB de casos (clave de partición: id, tipo String)
  AWS_REGION       región de AWS (por defecto us-east-1)
  PANEL_CLAVE      clave para entrar al panel de casos (si no se define, el panel queda abierto)
  SECRET_KEY       clave para firmar las sesiones del panel
  MODELO_VERSION   versión del modelo que queda registrada en cada caso
"""
import io
import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps

import numpy as np
from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE, "modelo", "modelo_danos.tflite")
RUTA_ETIQUETAS = os.path.join(BASE, "modelo", "etiquetas.json")
RUTA_COSTOS = os.path.join(BASE, "costos.json")

BUCKET = os.environ.get("BUCKET_S3", "").strip()
TABLA = os.environ.get("TABLA_DYNAMO", "").strip()
REGION = os.environ.get("AWS_REGION", "us-east-1")
PANEL_CLAVE = os.environ.get("PANEL_CLAVE", "")
MODELO_VERSION_ENV = os.environ.get("MODELO_VERSION", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("siniestros")


def evento(nombre, **datos):
    """Log estructurado (una línea JSON por evento): fácil de buscar en journalctl o CloudWatch."""
    log.info(json.dumps({"evento": nombre, **datos}, ensure_ascii=False, default=str))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB por foto
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
if not PANEL_CLAVE:
    log.warning("PANEL_CLAVE no definida: el panel de casos queda sin clave")


@app.errorhandler(413)
def demasiado_grande(_):
    return jsonify(error="La foto pesa más de 10 MB. Prueba con otra o tómala con menor resolución."), 413


@app.errorhandler(500)
def error_interno(e):
    evento("error_interno", ruta=request.path, error=str(getattr(e, "original_exception", e))[:300])
    if request.path.startswith("/api/"):
        return jsonify(error="Ocurrió un error en el servidor. Intenta de nuevo en un momento."), 500
    return "Ocurrió un error en el servidor.", 500


@app.after_request
def cabeceras_seguridad(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    return resp


# ---------------------------------------------------------------- modelo
def _crear_interprete(ruta):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    interp = Interpreter(model_path=ruta)
    interp.allocate_tensors()
    return interp


with open(RUTA_ETIQUETAS, encoding="utf-8") as f:
    META = json.load(f)
with open(RUTA_COSTOS, encoding="utf-8") as f:
    COSTOS = json.load(f)

CLASES = META["clases"]
# La versión viene del modelo exportado; la variable de entorno solo la sobrescribe si se define
MODELO_VERSION = META.get("version") or MODELO_VERSION_ENV or "efficientnetb0-v2"
SEVERIDADES = list(COSTOS["factor_severidad"])
INTERP = _crear_interprete(RUTA_MODELO)
ENTRADA = INTERP.get_input_details()[0]
SALIDA = INTERP.get_output_details()[0]
ALTO, ANCHO = int(ENTRADA["shape"][1]), int(ENTRADA["shape"][2])


def predecir(imagen: Image.Image):
    """Devuelve {clase: probabilidad} para una imagen PIL."""
    img = ImageOps.exif_transpose(imagen).convert("RGB").resize((ANCHO, ALTO), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)[None, ...]  # EfficientNet espera píxeles 0-255
    INTERP.set_tensor(ENTRADA["index"], x)
    INTERP.invoke()
    p = INTERP.get_tensor(SALIDA["index"])[0].astype(float)
    return {c: float(v) for c, v in zip(CLASES, p)}


# ---------------------------------------------------------------- negocio
def severidad_de(clase):
    return COSTOS["mapeo_etiquetas"].get(clase, clase)


def costo_para(severidad, pieza, vehiculo):
    base = COSTOS["costo_base_clp"][pieza]
    f_sev = COSTOS["factor_severidad"].get(severidad, 1.0)
    f_veh = COSTOS["factor_vehiculo"][vehiculo]
    return base * f_sev * f_veh


def evaluar(probs, pieza, vehiculo, deducible, valor_comercial):
    clase = max(probs, key=probs.get)
    confianza = probs[clase]
    severidad = severidad_de(clase)

    costo = costo_para(severidad, pieza, vehiculo)
    # Costo esperado: costo de cada severidad ponderado por su probabilidad
    esperado = sum(p * costo_para(severidad_de(c), pieza, vehiculo) for c, p in probs.items())
    m = COSTOS["margen_estimacion"]

    if confianza < COSTOS["umbral_confianza"]:
        decision, nivel, codigo = "Revisión por un liquidador: el modelo no está seguro de la severidad", "revision", "revision"
    elif valor_comercial and esperado >= COSTOS["umbral_perdida_total"] * valor_comercial:
        decision, nivel, codigo = "Posible pérdida total: derivar a liquidador presencial", "grave", "perdida_total"
    elif severidad == "grave":
        decision, nivel, codigo = "Daño grave: requiere inspección presencial antes de aprobar", "grave", "inspeccion"
    elif deducible and esperado <= deducible:
        decision, nivel, codigo = "El costo estimado no supera el deducible: probablemente no conviene reclamar", "info", "bajo_deducible"
    else:
        decision, nivel, codigo = "Aprobación rápida: derivar a taller en convenio", "ok", "aprobacion"

    return {
        "clase": clase,
        "severidad": severidad,
        "confianza": round(confianza, 4),
        "probabilidades": {severidad_de(c): round(p, 4) for c, p in probs.items()},
        "costo_estimado": round(costo),
        "rango": [round(costo * (1 - m)), round(costo * (1 + m))],
        "costo_esperado": round(esperado),
        "decision": decision,
        "decision_codigo": codigo,
        "nivel": nivel,
        "estado": "pendiente_revision" if nivel in ("revision", "grave") else "resuelto_automatico",
        "moneda": COSTOS["moneda"],
    }


# ---------------------------------------------------------------- almacenamiento
def _a_python(v):
    """DynamoDB devuelve Decimal; se convierte a int/float para JSON."""
    if isinstance(v, list):
        return [_a_python(x) for x in v]
    if isinstance(v, dict):
        return {k: _a_python(x) for k, x in v.items()}
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    return v


class Almacen:
    """Fotos en S3 y casos en DynamoDB; si no están configurados, en memoria (solo para pruebas)."""

    def __init__(self):
        self.fotos_mem, self.casos_mem = {}, {}
        self.s3 = self.tabla = None
        if BUCKET or TABLA:
            import boto3
            from botocore.config import Config
            if BUCKET:
                self.s3 = boto3.client("s3", region_name=REGION, endpoint_url=f"https://s3.{REGION}.amazonaws.com", config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
            if TABLA:
                self.tabla = boto3.resource("dynamodb", region_name=REGION).Table(TABLA)

    # fotos
    def guardar_foto(self, id_caso, datos, extension, tipo):
        if self.s3:
            llave = f"siniestros/{id_caso}{extension}"
            self.s3.put_object(Bucket=BUCKET, Key=llave, Body=datos, ContentType=tipo,
                               ServerSideEncryption="AES256")
            return f"s3://{BUCKET}/{llave}"
        self.fotos_mem[id_caso] = (datos, tipo)
        return f"memoria://{id_caso}"

    def url_foto(self, caso):
        ref = caso.get("foto_ref", "")
        if ref.startswith("s3://") and self.s3:
            llave = ref.split("/", 3)[3]
            return self.s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": llave}, ExpiresIn=900)
        if ref.startswith("memoria://"):
            return url_for("foto_memoria", id_caso=caso["id"])
        return ""

    def leer_foto(self, caso):
        """Devuelve (bytes, extensión) de la foto del caso."""
        ref = caso.get("foto_ref", "")
        if ref.startswith("s3://") and self.s3:
            llave = ref.split("/", 3)[3]
            return self.s3.get_object(Bucket=BUCKET, Key=llave)["Body"].read(), os.path.splitext(llave)[1]
        if ref.startswith("memoria://") and caso["id"] in self.fotos_mem:
            datos, tipo = self.fotos_mem[caso["id"]]
            return datos, ".png" if "png" in tipo else ".jpg"
        return None, ""

    # casos
    def guardar_caso(self, caso):
        if self.tabla:
            self.tabla.put_item(Item=json.loads(json.dumps(caso), parse_float=Decimal))
        else:
            self.casos_mem[caso["id"]] = caso

    def listar_casos(self, maximo=1000):
        if not self.tabla:
            casos = list(self.casos_mem.values())
        else:
            casos, kwargs = [], {}
            while len(casos) < maximo:
                r = self.tabla.scan(**kwargs)
                casos += r.get("Items", [])
                if "LastEvaluatedKey" not in r:
                    break
                kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]
            casos = _a_python(casos)
        return sorted(casos, key=lambda c: c.get("fecha", ""), reverse=True)

    def obtener_caso(self, id_caso):
        if self.tabla:
            return _a_python(self.tabla.get_item(Key={"id": id_caso}).get("Item"))
        return self.casos_mem.get(id_caso)

    def guardar_revision(self, id_caso, campos):
        if self.tabla:
            nombres = {f"#{k}": k for k in campos}
            valores = {f":{k}": v for k, v in json.loads(json.dumps(campos), parse_float=Decimal).items()}
            self.tabla.update_item(
                Key={"id": id_caso},
                UpdateExpression="SET " + ", ".join(f"#{k} = :{k}" for k in campos),
                ExpressionAttributeNames=nombres, ExpressionAttributeValues=valores,
                ConditionExpression="attribute_exists(id)")
        else:
            self.casos_mem[id_caso].update(campos)


ALMACEN = Almacen()


# ---------------------------------------------------------------- acceso al panel
def requiere_login(vista):
    @wraps(vista)
    def envoltura(*args, **kwargs):
        if PANEL_CLAVE and not session.get("liquidador"):
            if request.path.startswith("/api/"):
                return jsonify(error="Sesión requerida"), 401
            return redirect(url_for("login", siguiente=request.path))
        return vista(*args, **kwargs)
    return envoltura


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        nombre = (request.form.get("nombre") or "").strip()[:60]
        if nombre and secrets.compare_digest(request.form.get("clave", ""), PANEL_CLAVE):
            session["liquidador"] = nombre
            evento("login", liquidador=nombre)
            siguiente = request.args.get("siguiente", "/casos")
            return redirect(siguiente if siguiente.startswith("/") else "/casos")
        error = "La clave no es correcta. Revísala e intenta de nuevo."
        evento("login_fallido", ip=request.headers.get("X-Forwarded-For", request.remote_addr))
    return render_template("login.html", error=error, interno=True)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("inicio"))


# ---------------------------------------------------------------- páginas
@app.get("/")
def inicio():
    return render_template(
        "index.html",
        piezas=list(COSTOS["costo_base_clp"]),
        vehiculos=list(COSTOS["factor_vehiculo"]),
        metricas=META.get("metricas_test", {}),
    )


@app.get("/casos")
@requiere_login
def panel_casos():
    return render_template("casos.html", severidades=SEVERIDADES, interno=True,
                           piezas=list(COSTOS["costo_base_clp"]), vehiculos=list(COSTOS["factor_vehiculo"]),
                           motivos=MOTIVOS_DESCARTE)


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo": MODELO_VERSION, "clases": CLASES,
            "s3": bool(BUCKET), "dynamodb": bool(TABLA)}


# ---------------------------------------------------------------- API
def _numero(nombre):
    valor = (request.form.get(nombre) or "").replace(".", "").replace(",", "").strip()
    return float(valor) if valor else 0.0


@app.post("/api/evaluar")
def api_evaluar():
    t0 = time.perf_counter()
    foto = request.files.get("foto")
    if not foto or not foto.filename:
        return jsonify(error="Falta la foto"), 400
    pieza = request.form.get("pieza", "")
    vehiculo = request.form.get("vehiculo", "")
    if pieza not in COSTOS["costo_base_clp"] or vehiculo not in COSTOS["factor_vehiculo"]:
        return jsonify(error="Pieza o tipo de vehículo no válido"), 400
    try:
        deducible, valor = _numero("deducible"), _numero("valor_comercial")
    except ValueError:
        return jsonify(error="Deducible y valor comercial deben ser números"), 400

    datos = foto.read()
    try:
        imagen = Image.open(io.BytesIO(datos))
        imagen.load()
    except Exception:
        return jsonify(error="El archivo no es una imagen válida"), 400

    caso = evaluar(predecir(imagen), pieza, vehiculo, deducible, valor)
    caso.update({
        "id": uuid.uuid4().hex[:12],
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pieza": pieza,
        "vehiculo": vehiculo,
        "deducible": deducible,
        "valor_comercial": valor,
        "modelo_version": MODELO_VERSION,
    })
    ext = os.path.splitext(foto.filename)[1].lower()
    ext = ext if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"
    tipo = Image.MIME.get(imagen.format, "image/jpeg")

    try:
        caso["foto_ref"] = ALMACEN.guardar_foto(caso["id"], datos, ext, tipo)
        ALMACEN.guardar_caso(caso)
        caso["guardado"] = True
    except Exception as e:  # la evaluación se entrega igual aunque falle el guardado
        caso["guardado"] = False
        evento("error_guardado", id=caso["id"], error=e.__class__.__name__, detalle=str(e)[:200])

    caso["latencia_ms"] = round((time.perf_counter() - t0) * 1000)
    evento("evaluacion", id=caso["id"], severidad=caso["severidad"], confianza=caso["confianza"],
           estado=caso["estado"], latencia_ms=caso["latencia_ms"], guardado=caso["guardado"])
    return jsonify(caso)


@app.get("/api/casos")
@requiere_login
def api_casos():
    estado = request.args.get("estado", "")
    casos = [dict(c) for c in ALMACEN.listar_casos() if not estado or c.get("estado") == estado][:200]
    for c in casos:
        try:
            c["foto_url"] = ALMACEN.url_foto(c)
        except Exception:
            c["foto_url"] = ""
    return jsonify(casos=casos)


@app.get("/api/casos/<id_caso>")
@requiere_login
def api_caso(id_caso):
    caso = ALMACEN.obtener_caso(id_caso)
    if not caso:
        return jsonify(error="Caso no encontrado"), 404
    caso = dict(caso)
    caso["foto_url"] = ALMACEN.url_foto(caso)
    return jsonify(caso)


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _con_historial(caso, accion, detalle, campos):
    """Agrega una entrada al historial de auditoría del caso y la incluye en los campos a guardar."""
    historial = list(caso.get("historial") or [])
    historial.append({"fecha": _ahora(), "usuario": session.get("liquidador", "sin sesión"),
                      "accion": accion, "detalle": detalle})
    campos["historial"] = historial[-50:]
    return campos


MOTIVOS_DESCARTE = {
    "datos_invalidos": "Datos inventados o de prueba",
    "foto_no_valida": "La foto no muestra un daño vehicular",
    "duplicado": "Caso duplicado",
    "otro": "Otro motivo",
}


@app.post("/api/casos/<id_caso>/revision")
@requiere_login
def api_revision(id_caso):
    caso = ALMACEN.obtener_caso(id_caso)
    if not caso:
        return jsonify(error="Caso no encontrado"), 404
    if caso.get("estado") == "descartado":
        return jsonify(error="Restaura el caso antes de revisarlo"), 409
    datos = request.get_json(silent=True) or {}
    sev = datos.get("severidad")
    if sev not in SEVERIDADES:
        return jsonify(error="Severidad no válida"), 400
    campos = {
        "estado": "revisado",
        "severidad_revisada": sev,
        "coincide": sev == caso["severidad"],
        "costo_revisado": round(costo_para(sev, caso["pieza"], caso["vehiculo"])),
        "comentario": str(datos.get("comentario", ""))[:500],
        "revisor": session.get("liquidador", "sin sesión"),
        "fecha_revision": _ahora(),
    }
    _con_historial(caso, "revision", f"Gravedad confirmada: {sev}", campos)
    ALMACEN.guardar_revision(id_caso, campos)
    evento("revision", id=id_caso, modelo=caso["severidad"], liquidador=sev,
           coincide=campos["coincide"], revisor=campos["revisor"])
    return jsonify({**caso, **campos})


@app.post("/api/casos/<id_caso>/datos")
@requiere_login
def api_corregir_datos(id_caso):
    """Corrige pieza, vehículo, deducible o valor y recalcula costo y decisión con las mismas probabilidades."""
    caso = ALMACEN.obtener_caso(id_caso)
    if not caso:
        return jsonify(error="Caso no encontrado"), 404
    d = request.get_json(silent=True) or {}
    pieza, vehiculo = d.get("pieza"), d.get("vehiculo")
    motivo = str(d.get("motivo", "")).strip()[:300]
    if pieza not in COSTOS["costo_base_clp"] or vehiculo not in COSTOS["factor_vehiculo"]:
        return jsonify(error="Pieza o tipo de vehículo no válido"), 400
    if not motivo:
        return jsonify(error="Indica el motivo de la corrección"), 400
    try:
        deducible = float(str(d.get("deducible") or 0).replace(".", ""))
        valor = float(str(d.get("valor_comercial") or 0).replace(".", ""))
    except ValueError:
        return jsonify(error="Deducible y valor deben ser números"), 400

    nuevo = evaluar(caso["probabilidades"], pieza, vehiculo, deducible, valor)
    def fmt(v):
        return f"${int(v or 0):,}".replace(",", ".") if isinstance(v, (int, float)) else str(v)
    cambios = [f"{n}: {fmt(caso.get(k))} → {fmt(v)}" for n, k, v in [
        ("parte", "pieza", pieza), ("vehículo", "vehiculo", vehiculo),
        ("deducible", "deducible", deducible), ("valor", "valor_comercial", valor)] if caso.get(k) != v]
    if not cambios:
        return jsonify(error="No hay cambios que guardar"), 400
    campos = {
        "pieza": pieza, "vehiculo": vehiculo, "deducible": deducible, "valor_comercial": valor,
        "costo_estimado": nuevo["costo_estimado"], "rango": nuevo["rango"], "costo_esperado": nuevo["costo_esperado"],
        "decision": nuevo["decision"], "decision_codigo": nuevo["decision_codigo"], "nivel": nuevo["nivel"],
    }
    if not caso.get("datos_originales"):
        campos["datos_originales"] = {k: caso.get(k) for k in ("pieza", "vehiculo", "deducible", "valor_comercial")}
    if caso.get("estado") == "revisado":
        campos["costo_revisado"] = round(costo_para(caso["severidad_revisada"], pieza, vehiculo))
    elif caso.get("estado") != "descartado":
        campos["estado"] = nuevo["estado"]
    _con_historial(caso, "correccion", "; ".join(cambios) + f". Motivo: {motivo}", campos)
    ALMACEN.guardar_revision(id_caso, campos)
    evento("correccion_datos", id=id_caso, cambios=cambios, usuario=session.get("liquidador"))
    return jsonify({**caso, **campos})


@app.post("/api/casos/<id_caso>/descartar")
@requiere_login
def api_descartar(id_caso):
    caso = ALMACEN.obtener_caso(id_caso)
    if not caso:
        return jsonify(error="Caso no encontrado"), 404
    if caso.get("estado") == "descartado":
        return jsonify(error="El caso ya está descartado"), 409
    d = request.get_json(silent=True) or {}
    motivo = d.get("motivo")
    if motivo not in MOTIVOS_DESCARTE:
        return jsonify(error="Elige un motivo de descarte"), 400
    nota = str(d.get("nota", "")).strip()[:300]
    if motivo == "otro" and not nota:
        return jsonify(error="Describe el motivo"), 400
    campos = {
        "estado": "descartado", "estado_previo": caso.get("estado"),
        "motivo_descarte": motivo, "nota_descarte": nota,
        "descartado_por": session.get("liquidador", "sin sesión"), "fecha_descarte": _ahora(),
    }
    _con_historial(caso, "descarte", MOTIVOS_DESCARTE[motivo] + (f": {nota}" if nota else ""), campos)
    ALMACEN.guardar_revision(id_caso, campos)
    evento("descarte", id=id_caso, motivo=motivo, usuario=campos["descartado_por"])
    return jsonify({**caso, **campos})


@app.post("/api/casos/<id_caso>/restaurar")
@requiere_login
def api_restaurar(id_caso):
    caso = ALMACEN.obtener_caso(id_caso)
    if not caso:
        return jsonify(error="Caso no encontrado"), 404
    if caso.get("estado") != "descartado":
        return jsonify(error="El caso no está descartado"), 409
    campos = {"estado": caso.get("estado_previo") or "pendiente_revision"}
    _con_historial(caso, "restauracion", "Caso restaurado", campos)
    ALMACEN.guardar_revision(id_caso, campos)
    evento("restauracion", id=id_caso, usuario=session.get("liquidador"))
    return jsonify({**caso, **campos})


@app.get("/api/entrenamiento.csv")
@requiere_login
def api_entrenamiento():
    """Casos revisados por un liquidador y no descartados: el set limpio para reentrenar el modelo."""
    import csv
    salida = io.StringIO()
    w = csv.writer(salida)
    w.writerow(["id", "fecha", "foto_ref", "pieza", "vehiculo", "severidad_modelo",
                "severidad_liquidador", "confianza", "modelo_version", "revisor"])
    n = 0
    for c in ALMACEN.listar_casos():
        if c.get("estado") == "revisado" and c.get("foto_ref"):
            w.writerow([c["id"], c.get("fecha"), c["foto_ref"], c.get("pieza"), c.get("vehiculo"),
                        c.get("severidad"), c.get("severidad_revisada"), c.get("confianza"),
                        c.get("modelo_version"), c.get("revisor")])
            n += 1
    evento("exportacion_entrenamiento", filas=n, usuario=session.get("liquidador"))
    nombre = f"entrenamiento_{datetime.now(timezone.utc):%Y%m%d}.csv"
    return app.response_class(salida.getvalue().encode("utf-8-sig"), mimetype="text/csv",
                              headers={"Content-Disposition": f"attachment; filename={nombre}"})


@app.get("/api/entrenamiento.zip")
@requiere_login
def api_entrenamiento_zip():
    """Fotos de los casos revisados y no descartados, en carpetas por gravedad confirmada, más el CSV."""
    import csv
    import zipfile
    buf, tabla = io.BytesIO(), io.StringIO()
    w = csv.writer(tabla)
    w.writerow(["id", "fecha", "archivo", "pieza", "vehiculo", "severidad_modelo",
                "severidad_liquidador", "confianza", "modelo_version", "revisor"])
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for c in ALMACEN.listar_casos():
            if c.get("estado") != "revisado":
                continue
            try:
                datos, ext = ALMACEN.leer_foto(c)
            except Exception as e:
                evento("error_exportacion", id=c["id"], error=e.__class__.__name__)
                continue
            if not datos:
                continue
            archivo = f"fotos/{c['severidad_revisada']}/{c['id']}{ext or '.jpg'}"
            z.writestr(archivo, datos)
            w.writerow([c["id"], c.get("fecha"), archivo, c.get("pieza"), c.get("vehiculo"), c.get("severidad"),
                        c.get("severidad_revisada"), c.get("confianza"), c.get("modelo_version"), c.get("revisor")])
            n += 1
        z.writestr("entrenamiento.csv", "\ufeff" + tabla.getvalue())
    evento("exportacion_entrenamiento_zip", casos=n, usuario=session.get("liquidador"))
    nombre = f"casos_revisados_{datetime.now(timezone.utc):%Y%m%d}.zip"
    return app.response_class(buf.getvalue(), mimetype="application/zip",
                              headers={"Content-Disposition": f"attachment; filename={nombre}"})


@app.get("/api/metricas")
@requiere_login
def api_metricas():
    todos = ALMACEN.listar_casos()
    casos = [c for c in todos if c.get("estado") != "descartado"]  # los descartados no cuentan
    total = len(casos)
    auto = sum(c.get("estado") == "resuelto_automatico" for c in casos)
    pendientes = sum(c.get("estado") == "pendiente_revision" for c in casos)
    revisados = [c for c in casos if c.get("estado") == "revisado"]
    coinciden = sum(bool(c.get("coincide")) for c in revisados)
    return jsonify({
        "total": total,
        "resueltos_automaticos": auto,
        "pendientes": pendientes,
        "revisados": len(revisados),
        "descartados": len(todos) - total,
        "tasa_automatizacion": round(auto / total, 4) if total else None,
        "concordancia_liquidador": round(coinciden / len(revisados), 4) if revisados else None,
        "confianza_promedio": round(sum(c.get("confianza", 0) for c in casos) / total, 4) if total else None,
        "costo_estimado_total": sum(c.get("costo_revisado", c.get("costo_estimado", 0)) for c in casos),
        "por_severidad": {s: sum(c.get("severidad") == s for c in casos) for s in SEVERIDADES},
        "modelo": MODELO_VERSION,
        "metricas_modelo": META.get("metricas_test", {}),
    })


@app.get("/foto/<id_caso>")
@requiere_login
def foto_memoria(id_caso):
    if id_caso not in ALMACEN.fotos_mem:
        abort(404)
    datos, tipo = ALMACEN.fotos_mem[id_caso]
    return send_file(io.BytesIO(datos), mimetype=tipo)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PUERTO", 5000)))
