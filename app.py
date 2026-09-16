# -*- coding: utf-8 -*-
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from chicharras_core import (
    cargar_audio, validar_intervalo, extraer_segmento, remover_dc,
    filtro_pasabanda, reducir_ruido, calcular_metricas_temporales,
    calcular_sonograma, calcular_fft, calcular_envolvente_espectral,
    detectar_maximos_espectrales, frecuencia_dominante_vs_tiempo,
    detectar_pulsos, calcular_modulacion_frecuencia,
    envolvente_temporal_amplitud, calcular_modulacion_amplitud,
    centroide_y_ancho_banda,
    graficar_forma_onda, graficar_forma_onda_comparativa,
    graficar_sonograma, graficar_fft, graficar_comparacion_fft,
    graficar_comparacion_maximos_espectrales,
    graficar_frecuencia_vs_tiempo, graficar_frecuencia_vs_tiempo_comparativa,
    graficar_comparacion_frecuencia_ventanas, graficar_envolvente_pulsos,
    graficar_comparacion_pulsos, graficar_comparacion_rms,
    graficar_distribucion_frecuencia_dominante, graficar_distribucion_rms,
)

st.set_page_config(
    page_title="Análisis acústico de chicharras",
    page_icon="🦗",
    layout="wide",
)

st.title("🦗 Análisis acústico de chicharras")
st.caption("Aplicación web basada en el programa de análisis desarrollado originalmente en Google Colab.")

with st.sidebar:
    st.header("1. Archivos de audio")
    archivo_manana = st.file_uploader(
        "Grabación de la mañana", type=["wav", "mp3", "flac", "ogg", "m4a"], key="manana"
    )
    archivo_tarde = st.file_uploader(
        "Grabación de la tarde", type=["wav", "mp3", "flac", "ogg", "m4a"], key="tarde"
    )

    st.header("2. Intervalos")
    inicio_manana = st.number_input("Inicio mañana (s)", min_value=0.0, value=10.0, step=1.0)
    final_manana = st.number_input("Final mañana (s)", min_value=0.01, value=40.0, step=1.0)
    inicio_tarde = st.number_input("Inicio tarde (s)", min_value=0.0, value=300.0, step=1.0)
    final_tarde = st.number_input("Final tarde (s)", min_value=0.01, value=310.0, step=1.0)

    st.header("3. Procesamiento")
    fmin = st.number_input("Frecuencia mínima (Hz)", min_value=1.0, value=1000.0, step=100.0)
    fmax = st.number_input("Frecuencia máxima (Hz)", min_value=10.0, value=12000.0, step=100.0)
    orden = st.slider("Orden del filtro", 1, 10, 4)
    usar_ruido = st.checkbox("Aplicar reducción de ruido", value=True)

    st.header("4. Análisis")
    ventana = st.number_input("Ventana de análisis (s)", min_value=0.01, value=10.0, step=1.0)
    prom = st.number_input("Prominencia de picos (dB)", min_value=0.0, value=3.0, step=0.5)
    dist = st.number_input("Distancia mínima entre picos (bins)", min_value=1, value=20, step=1)
    nmax = st.number_input("Máximos espectrales a reportar", min_value=1, max_value=30, value=8, step=1)

    st.header("5. Visualización")
    frecuencia_sonograma_max = st.number_input(
        "Frecuencia máxima del sonograma (Hz)", min_value=100.0, value=15000.0, step=500.0
    )

if not archivo_manana or not archivo_tarde:
    st.info("Carga las dos grabaciones para comenzar. La aplicación analiza mañana y tarde y permite compararlas.")
    st.stop()

def cargar_desde_upload(upload):
    return cargar_audio(upload)

@st.cache_data(show_spinner=False)
def ejecutar_analisis(
    bytes_manana, bytes_tarde,
    ini_m, fin_m, ini_t, fin_t,
    fmin_, fmax_, orden_, ruido_, ventana_, prom_, dist_, nmax_, fson_
):
    from io import BytesIO
    y_m, sr_m, canales_m = cargar_audio(BytesIO(bytes_manana))
    y_t, sr_t, canales_t = cargar_audio(BytesIO(bytes_tarde))

    audios = {
        "manana": {"y": y_m, "sr": sr_m, "canales": canales_m},
        "tarde": {"y": y_t, "sr": sr_t, "canales": canales_t},
    }
    intervalos = {"manana": (ini_m, fin_m), "tarde": (ini_t, fin_t)}
    segmentos = {}
    procesados = {}
    resultados = {}

    for s in ("manana", "tarde"):
        y = audios[s]["y"]
        sr = audios[s]["sr"]
        dur = len(y) / sr
        ini, fin = intervalos[s]
        fin = validar_intervalo(ini, fin, dur, s)
        intervalos[s] = (ini, fin)
        seg = extraer_segmento(y, sr, ini, fin)
        segmentos[s] = {"y": seg, "sr": sr}

        y0 = remover_dc(seg)
        yf = filtro_pasabanda(y0, sr, fmin_, fmax_, orden_)
        yp = reducir_ruido(yf, sr, ruido_)
        procesados[s] = {"original": seg, "procesado": yp, "sr": sr}

        metricas = calcular_metricas_temporales(yp, sr)
        f_stft, t_stft, Sxx = calcular_sonograma(yp, sr, 1024, 768)
        freqs, magdb = calcular_fft(yp, sr)
        envesp = calcular_envolvente_espectral(freqs, magdb, 200)
        dfp, fdom = detectar_maximos_espectrales(freqs, magdb, prom_, dist_, None)
        centroide, ancho = centroide_y_ancho_banda(freqs, magdb)
        tf, fdt, stats = frecuencia_dominante_vs_tiempo(yp, sr, 1024, 768)
        pulsos = detectar_pulsos(yp, sr, 0.02, 0.02, None)
        modf = calcular_modulacion_frecuencia(fdt)
        env = envolvente_temporal_amplitud(yp)
        moda = calcular_modulacion_amplitud(env, sr)

        resultados[s] = {
            "y_original": seg, "y_procesado": yp, "sr": sr,
            "metricas": metricas,
            "sonograma": {"f": f_stft, "t": t_stft, "Sxx_db": Sxx},
            "espectro": {
                "freqs": freqs, "magnitud_db": magdb, "envolvente": envesp,
                "centroide": centroide, "ancho_banda": ancho,
            },
            "maximos": {"df_picos": dfp, "f_dominante": fdom},
            "frecuencia_vs_tiempo": {"t": tf, "f_dom_t": fdt, "stats": stats},
            "pulsos": pulsos,
            "modulacion_frecuencia": modf,
            "modulacion_amplitud": moda,
            "envolvente_temporal": env,
        }

    filas_segmentos = []
    filas_maximos = []
    for s in ("manana", "tarde"):
        yint = segmentos[s]["y"]
        sr = segmentos[s]["sr"]
        ini, fin = intervalos[s]
        nwin = max(1, int(ventana_ * sr))
        for i, i0 in enumerate(range(0, len(yint), nwin)):
            i1 = min(i0 + nwin, len(yint))
            yw = yint[i0:i1]
            if len(yw) < 8:
                continue
            fila, dfp = analizar_ventana_web(yw, sr, fmin_, fmax_, orden_, ruido_, prom_, dist_)
            fila_full = {
                "sesion": "Mañana" if s == "manana" else "Tarde",
                "segmento": i,
                "inicio_s": ini + i0 / sr,
                "final_s": ini + i1 / sr,
                "duracion_s": (i1 - i0) / sr,
            }
            fila_full.update(fila)
            filas_segmentos.append(fila_full)
            if len(dfp):
                for _, pico in dfp.head(nmax_).iterrows():
                    filas_maximos.append({
                        "sesion": "Mañana" if s == "manana" else "Tarde",
                        "segmento": i,
                        "inicio_s": ini + i0 / sr,
                        "final_s": ini + i1 / sr,
                        "numero_pico": int(pico["numero_pico"]),
                        "frecuencia_hz": pico["frecuencia_hz"],
                        "magnitud": pico["magnitud_db"],
                        "prominencia": pico["prominencia"],
                        "ancho_pico": pico.get("ancho_pico_hz", np.nan),
                    })

    tabla_resultados = pd.DataFrame(filas_segmentos)
    tabla_maximos = pd.DataFrame(filas_maximos)

    columnas = [
        "rms", "energia", "amplitud_max", "frecuencia_dominante",
        "centroide_espectral", "ancho_banda", "tasa_pulsos",
        "modulacion_frecuencia", "modulacion_amplitud"
    ]
    resumen = []
    for col in columnas:
        for etiqueta in ("Mañana", "Tarde"):
            if col not in tabla_resultados:
                continue
            d = tabla_resultados.loc[tabla_resultados["sesion"] == etiqueta, col].dropna()
            if len(d):
                resumen.append({
                    "parametro": col, "sesion": etiqueta,
                    "media": d.mean(), "mediana": d.median(), "std": d.std(),
                    "min": d.min(), "max": d.max(),
                    "Q1": d.quantile(.25), "Q3": d.quantile(.75),
                    "IQR": d.quantile(.75) - d.quantile(.25),
                })
    tabla_comparacion = pd.DataFrame(resumen)
    if len(tabla_comparacion):
        resumen_comparativo = tabla_comparacion.pivot(
            index="parametro", columns="sesion", values="media"
        )
    else:
        resumen_comparativo = pd.DataFrame()

    return audios, intervalos, segmentos, procesados, resultados, tabla_resultados, tabla_maximos, tabla_comparacion, resumen_comparativo

def analizar_ventana_web(y_ventana, sr, fmin_, fmax_, orden_, ruido_, prom_, dist_):
    y0 = remover_dc(y_ventana)
    yf = filtro_pasabanda(y0, sr, fmin_, fmax_, orden_)
    yp = reducir_ruido(yf, sr, ruido_)
    met = calcular_metricas_temporales(yp, sr)
    freqs, magdb = calcular_fft(yp, sr)
    dfp, fdom = detectar_maximos_espectrales(freqs, magdb, prom_, dist_, None)
    cen, ancho = centroide_y_ancho_banda(freqs, magdb)
    pulsos = detectar_pulsos(yp, sr, 0.02, 0.02, None)
    tasa = pulsos["tasa_pulsos_por_s"] if pulsos else np.nan
    try:
        nloc = min(1024, len(yp))
        oloc = min(768, max(0, nloc - 1))
        _, fdt, _ = frecuencia_dominante_vs_tiempo(yp, sr, nloc, oloc)
        modf = calcular_modulacion_frecuencia(fdt) if len(fdt) > 2 else None
    except Exception:
        modf = None
    env = envolvente_temporal_amplitud(yp)
    moda = calcular_modulacion_amplitud(env, sr)
    fila = {
        "rms": met["rms"], "energia": met["energia"],
        "amplitud_max": met["amplitud_max"],
        "frecuencia_dominante": fdom if fdom is not None else np.nan,
        "centroide_espectral": cen, "ancho_banda": ancho,
        "tasa_pulsos": tasa,
        "modulacion_frecuencia": modf["rango_modulacion_hz"] if modf else np.nan,
        "modulacion_amplitud": moda["rango_amplitud"],
    }
    return fila, dfp

try:
    resultados = ejecutar_analisis(
        archivo_manana.getvalue(), archivo_tarde.getvalue(),
        inicio_manana, final_manana, inicio_tarde, final_tarde,
        fmin, fmax, orden, usar_ruido, ventana, prom, int(dist), int(nmax),
        frecuencia_sonograma_max
    )
except Exception as exc:
    st.error(f"No fue posible procesar las grabaciones: {exc}")
    st.stop()

audios, intervalos, segmentos, procesados, RESULTADOS, tabla_resultados, tabla_maximos, tabla_comparacion, resumen_comparativo = resultados

st.success("Análisis completado.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frecuencia dominante — mañana", f"{RESULTADOS['manana']['maximos']['f_dominante']:.1f} Hz" if RESULTADOS["manana"]["maximos"]["f_dominante"] is not None else "—")
c2.metric("Frecuencia dominante — tarde", f"{RESULTADOS['tarde']['maximos']['f_dominante']:.1f} Hz" if RESULTADOS["tarde"]["maximos"]["f_dominante"] is not None else "—")
c3.metric("Pulsos/s — mañana", f"{RESULTADOS['manana']['pulsos']['tasa_pulsos_por_s']:.2f}" if RESULTADOS["manana"]["pulsos"] else "—")
c4.metric("Pulsos/s — tarde", f"{RESULTADOS['tarde']['pulsos']['tasa_pulsos_por_s']:.2f}" if RESULTADOS["tarde"]["pulsos"] else "—")

st.subheader("Reproducción")
for s, label in (("manana", "Mañana"), ("tarde", "Tarde")):
    st.write(f"**{label} — intervalo {intervalos[s][0]:.2f}–{intervalos[s][1]:.2f} s**")
    audio_buf = io.BytesIO()
    import soundfile as sf
    sf.write(audio_buf, segmentos[s]["y"], segmentos[s]["sr"], format="WAV")
    st.audio(audio_buf.getvalue(), format="audio/wav")

st.subheader("Resultados")
modo = st.radio("Modo de visualización", ["Mañana", "Tarde", "Comparar mañana vs. tarde"], horizontal=True)
opciones = st.multiselect(
    "Selecciona qué visualizar",
    [
        "Señal temporal", "Sonograma", "FFT + envolvente espectral",
        "Máximos espectrales", "Envolvente temporal y pulsos",
        "Frecuencia dominante vs. tiempo", "Modulación de frecuencia",
        "Modulación de amplitud", "Resumen numérico"
    ],
    default=["FFT + envolvente espectral", "Resumen numérico"]
)

def mostrar_fig(fig, nombre):
    st.pyplot(fig, clear_figure=True)
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    st.download_button("⬇️ Descargar imagen", b.getvalue(), file_name=nombre, mime="image/png", key="img_"+nombre)

def mostrar_csv(df, nombre):
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=True).encode("utf-8"), file_name=nombre, mime="text/csv", key="csv_"+nombre)

for clave in opciones:
    if modo != "Comparar mañana vs. tarde":
        s = "manana" if modo == "Mañana" else "tarde"
        d = RESULTADOS[s]
        if clave == "Señal temporal":
            mostrar_fig(graficar_forma_onda(d["y_procesado"], d["sr"], f"Señal temporal — {modo}", "black"), f"{s}_senal.png")
        elif clave == "Sonograma":
            mostrar_fig(graficar_sonograma(d["sonograma"]["f"], d["sonograma"]["t"], d["sonograma"]["Sxx_db"], f"Sonograma — {modo}", frecuencia_sonograma_max), f"{s}_sonograma.png")
        elif clave == "FFT + envolvente espectral":
            mostrar_fig(graficar_fft(d["espectro"]["freqs"], d["espectro"]["magnitud_db"], d["espectro"]["envolvente"], None, f"FFT y envolvente — {modo}", mostrar_picos=False), f"{s}_fft.png")
        elif clave == "Máximos espectrales":
            mostrar_fig(graficar_fft(d["espectro"]["freqs"], d["espectro"]["magnitud_db"], d["espectro"]["envolvente"], d["maximos"]["df_picos"], f"Máximos espectrales — {modo}", n_top=int(nmax), mostrar_picos=True), f"{s}_maximos.png")
            if len(d["maximos"]["df_picos"]):
                mostrar_csv(d["maximos"]["df_picos"].head(int(nmax)), f"{s}_maximos.csv")
        elif clave == "Envolvente temporal y pulsos":
            mostrar_fig(graficar_envolvente_pulsos(d["y_procesado"], d["sr"], d["envolvente_temporal"], d["pulsos"], f"Envolvente temporal y pulsos — {modo}"), f"{s}_pulsos.png")
        elif clave == "Frecuencia dominante vs. tiempo":
            mostrar_fig(graficar_frecuencia_vs_tiempo(d["frecuencia_vs_tiempo"]["t"], d["frecuencia_vs_tiempo"]["f_dom_t"], f"Frecuencia dominante vs. tiempo — {modo}", "black"), f"{s}_frecuencia_tiempo.png")
        elif clave == "Modulación de frecuencia":
            mostrar_csv(pd.DataFrame([d["modulacion_frecuencia"]], index=[modo]), f"{s}_modulacion_frecuencia.csv")
        elif clave == "Modulación de amplitud":
            mostrar_csv(pd.DataFrame([d["modulacion_amplitud"]], index=[modo]), f"{s}_modulacion_amplitud.csv")
        elif clave == "Resumen numérico":
            fila = dict(d["metricas"])
            fila["frecuencia_dominante_hz"] = d["maximos"]["f_dominante"]
            fila["centroide_espectral_hz"] = d["espectro"]["centroide"]
            fila["ancho_banda_hz"] = d["espectro"]["ancho_banda"]
            fila["tasa_pulsos_por_s"] = d["pulsos"]["tasa_pulsos_por_s"] if d["pulsos"] else np.nan
            mostrar_csv(pd.DataFrame([fila], index=[modo]), f"{s}_resumen.csv")
    else:
        if clave == "Señal temporal":
            mostrar_fig(graficar_forma_onda_comparativa(procesados), "comparacion_senal.png")
        elif clave == "Sonograma":
            for s in ("manana", "tarde"):
                d = RESULTADOS[s]
                mostrar_fig(graficar_sonograma(d["sonograma"]["f"], d["sonograma"]["t"], d["sonograma"]["Sxx_db"], f"Sonograma — {ETIQUETAS[s]}", frecuencia_sonograma_max), f"{s}_sonograma.png")
        elif clave == "FFT + envolvente espectral":
            mostrar_fig(graficar_comparacion_fft(RESULTADOS), "comparacion_fft.png")
        elif clave == "Máximos espectrales":
            mostrar_fig(graficar_comparacion_maximos_espectrales(tabla_maximos), "comparacion_maximos.png")
        elif clave == "Envolvente temporal y pulsos":
            mostrar_fig(graficar_comparacion_pulsos(RESULTADOS), "comparacion_pulsos.png")
        elif clave == "Frecuencia dominante vs. tiempo":
            mostrar_fig(graficar_frecuencia_vs_tiempo_comparativa(RESULTADOS), "comparacion_frecuencia.png")
            if len(tabla_resultados):
                mostrar_fig(graficar_comparacion_frecuencia_ventanas(tabla_resultados), "comparacion_frecuencia_ventanas.png")
        elif clave == "Modulación de frecuencia":
            filas = [dict(RESULTADOS[s]["modulacion_frecuencia"], sesion=ETIQUETAS[s]) for s in ("manana", "tarde")]
            mostrar_csv(pd.DataFrame(filas).set_index("sesion"), "comparacion_modulacion_frecuencia.csv")
        elif clave == "Modulación de amplitud":
            filas = [dict(RESULTADOS[s]["modulacion_amplitud"], sesion=ETIQUETAS[s]) for s in ("manana", "tarde")]
            mostrar_csv(pd.DataFrame(filas).set_index("sesion"), "comparacion_modulacion_amplitud.csv")
        elif clave == "Resumen numérico":
            if len(tabla_comparacion):
                mostrar_csv(tabla_comparacion, "comparacion_estadistica_detallada.csv")
            if len(resumen_comparativo):
                mostrar_csv(resumen_comparativo, "resumen_comparativo.csv")
            if len(tabla_resultados):
                mostrar_fig(graficar_comparacion_rms(tabla_resultados), "comparacion_rms.png")
                mostrar_fig(graficar_distribucion_frecuencia_dominante(tabla_resultados), "distribucion_frecuencia.png")
                mostrar_fig(graficar_distribucion_rms(tabla_resultados), "distribucion_rms.png")

# Exportación conjunta
st.subheader("📦 Exportar resultados")
if st.button("Preparar ZIP con tablas y resultados"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        tablas = {
            "resultados_segmentos.csv": tabla_resultados,
            "maximos_espectrales.csv": tabla_maximos,
            "comparacion_estadistica_detallada.csv": tabla_comparacion,
            "resumen_comparativo.csv": resumen_comparativo,
        }
        for nombre, df in tablas.items():
            z.writestr(f"tablas/{nombre}", df.to_csv(index=True))
        for s in ("manana", "tarde"):
            d = RESULTADOS[s]
            figs = [
                (f"{s}_senal.png", graficar_forma_onda(d["y_procesado"], d["sr"], f"Señal temporal — {ETIQUETAS[s]}", "black")),
                (f"{s}_fft.png", graficar_fft(d["espectro"]["freqs"], d["espectro"]["magnitud_db"], d["espectro"]["envolvente"], d["maximos"]["df_picos"], f"FFT — {ETIQUETAS[s]}", n_top=int(nmax))),
            ]
            for nombre, fig in figs:
                b = io.BytesIO()
                fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
                z.writestr(f"figuras/{nombre}", b.getvalue())
                plt.close(fig)
    st.download_button(
        "⬇️ Descargar resultados_chicharras.zip",
        zip_buffer.getvalue(),
        file_name="resultados_chicharras.zip",
        mime="application/zip",
    )
