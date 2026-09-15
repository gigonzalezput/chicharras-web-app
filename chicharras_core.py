# -*- coding: utf-8 -*-
# Funciones científicas adaptadas desde analisis_chicharras_colab.py
# Esta versión es un módulo para la aplicación web Streamlit.

import os
import shutil
import zipfile
import warnings
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.signal import butter, sosfiltfilt, stft, find_peaks, hilbert
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

# Configuración por defecto del análisis
SESIONES = ["manana", "tarde"]
ETIQUETAS = {"manana": "Mañana", "tarde": "Tarde"}
colores = {"manana": "darkorange", "tarde": "steelblue"}

frecuencia_min = 1000
frecuencia_max = 12000
orden_filtro = 4
aplicar_reduccion_ruido = True
nperseg = 1024
noverlap = 768
frecuencia_max_sonograma = 15000
prominence = 3.0
distance = 20
height = None
n_maximos_reportar = 8
prominence_pulsos = 0.02
distance_pulsos_s = 0.02
height_pulsos = None
ventana_envolvente_hz = 200
def registrar_figura(fig, nombre_archivo):
    """Registra una figura en memoria para poder exportarla más tarde. NO la
    muestra en pantalla ni escribe nada en disco: la visualización queda a
    cargo de la interfaz interactiva (sección 12)."""
    if not any(nombre == nombre_archivo for nombre, _ in FIGURAS_PENDIENTES):
        FIGURAS_PENDIENTES.append((nombre_archivo, fig))

def registrar_tabla(df, nombre_archivo_csv):
    """Registra una tabla en memoria para poder exportarla más tarde. NO la
    muestra en pantalla."""
    TABLAS_PENDIENTES[nombre_archivo_csv] = df

def cargar_audio(path):
    """Carga un archivo de audio y lo convierte a mono si es necesario."""
    y, sr = librosa.load(path, sr=None, mono=False)
    canales = 1 if y.ndim == 1 else y.shape[0]
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y
    return y_mono, sr, canales

def mostrar_info(nombre_sesion, archivo, y, sr, canales):
    duracion = len(y) / sr
    print(f"--- {ETIQUETAS[nombre_sesion]} ---")
    print(f"  Archivo:            {archivo}")
    print(f"  Duración total:     {duracion:.2f} s")
    print(f"  Frecuencia muestreo:{sr} Hz")
    print(f"  Canales originales: {canales}")
    print(f"  N° de muestras:     {len(y)}")
    print()

def validar_intervalo(inicio, final, duracion, nombre_sesion):
    """Valida el intervalo configurado contra la duración real del audio
    cargado. Si el problema es irrecuperable (inicio negativo o inicio fuera
    de la grabación), lanza un error claro. Si solo el 'final' se pasa de la
    duración disponible, en vez de detener todo el notebook AJUSTA
    automáticamente el final al límite del archivo y avisa con una
    advertencia visible, para que el usuario note la discrepancia sin perder
    el resto del análisis."""
    errores = []
    if inicio < 0:
        errores.append("el inicio debe ser >= 0")
    if final <= inicio:
        errores.append("el final debe ser mayor que el inicio")
    if inicio >= duracion:
        errores.append(f"el inicio ({inicio}s) está fuera de la grabación, que solo dura {duracion:.2f}s")
    if errores:
        raise ValueError(
            f"Intervalo inválido para '{ETIQUETAS[nombre_sesion]}': " + "; ".join(errores) +
            ". Revisa que el archivo cargado sea realmente la grabación completa "
            "(verifica su duración con sf.info('archivo.wav') antes de continuar)."
        )

    if final > duracion:
        final_ajustado = duracion
        print(
            f"⚠️ ADVERTENCIA — '{ETIQUETAS[nombre_sesion]}': el final configurado ({final}s) supera la "
            f"duración real del archivo cargado ({duracion:.2f}s). Es posible que el archivo en Colab NO "
            f"sea la grabación completa (verifica con sf.info()). Se ajustó automáticamente el final a "
            f"{final_ajustado:.2f}s para poder continuar."
        )
        return final_ajustado
    return final

def extraer_segmento(y, sr, inicio, final):
    i0 = int(inicio * sr)
    i1 = int(final * sr)
    return y[i0:i1]

def remover_dc(y):
    return y - np.mean(y)

def filtro_pasabanda(y, sr, f_min, f_max, orden):
    nyq = sr / 2
    f_max_efectivo = min(f_max, nyq * 0.99)
    sos = butter(orden, [f_min, f_max_efectivo], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)

def reducir_ruido(y, sr, activar):
    if not activar:
        return y
    if NOISEREDUCE_DISPONIBLE:
        try:
            return nr.reduce_noise(y=y, sr=sr, stationary=False)
        except Exception:
            print("Advertencia: la reducción de ruido con 'noisereduce' falló; se usa la señal filtrada sin cambios adicionales.")
            return y
    else:
        print("Advertencia: librería 'noisereduce' no disponible; se omite este paso.")
        return y

def calcular_metricas_temporales(y, sr):
    duracion = len(y) / sr
    amp_max = float(np.max(y))
    amp_min = float(np.min(y))
    rms = float(np.sqrt(np.mean(y ** 2)))
    energia = float(np.sum(y ** 2))
    pico = max(abs(amp_max), abs(amp_min))
    factor_cresta = float(pico / rms) if rms > 0 else np.nan
    return {
        "duracion_s": duracion,
        "amplitud_max": amp_max,
        "amplitud_min": amp_min,
        "rms": rms,
        "energia": energia,
        "factor_cresta": factor_cresta,
    }

def calcular_sonograma(y, sr, nperseg, noverlap):
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    return f, t, Sxx_db

def calcular_fft(y, sr):
    """FFT de la señal completa del segmento. Devuelve frecuencias (Hz, solo
    parte positiva) y magnitud en dB."""
    n = len(y)
    X = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    magnitud = np.abs(X)
    magnitud_db = 20 * np.log10(magnitud + 1e-10)
    return freqs, magnitud_db

def calcular_envolvente_espectral(freqs, magnitud_db, ancho_hz):
    """Envolvente ESPECTRAL: suavizado de la magnitud de la FFT en función de
    la frecuencia (no confundir con la envolvente TEMPORAL de amplitud, que
    se calcula por separado mediante la transformada de Hilbert, más abajo).
    Se calcula con una media móvil cuyo ancho en Hz se convierte a número de
    muestras según la resolución espectral (freqs[1] - freqs[0])."""
    resolucion = freqs[1] - freqs[0] if len(freqs) > 1 else 1
    ventana_muestras = max(3, int(round(ancho_hz / resolucion)))
    if ventana_muestras % 2 == 0:
        ventana_muestras += 1
    kernel = np.ones(ventana_muestras) / ventana_muestras
    envolvente = np.convolve(magnitud_db, kernel, mode="same")
    return envolvente

def detectar_maximos_espectrales(freqs, magnitud_db, prominence, distance, height):
    """Detección CUANTITATIVA de máximos espectrales mediante
    scipy.signal.find_peaks. La frecuencia dominante es la del pico con
    mayor magnitud (en dB) entre los detectados."""
    picos_idx, propiedades = find_peaks(
        magnitud_db, prominence=prominence, distance=distance, height=height
    )
    if len(picos_idx) == 0:
        return pd.DataFrame(), None
    df_picos = pd.DataFrame({
        "indice": picos_idx,
        "frecuencia_hz": freqs[picos_idx],
        "magnitud_db": magnitud_db[picos_idx],
        "prominencia": propiedades.get("prominences", np.full(len(picos_idx), np.nan)),
    })
    anchos = sig.peak_widths(magnitud_db, picos_idx, rel_height=0.5)[0]
    resolucion = freqs[1] - freqs[0] if len(freqs) > 1 else 1
    df_picos["ancho_pico_hz"] = anchos * resolucion
    df_picos = df_picos.sort_values("magnitud_db", ascending=False).reset_index(drop=True)
    df_picos["numero_pico"] = df_picos.index + 1
    f_dominante = df_picos.iloc[0]["frecuencia_hz"]
    df_picos["frecuencia_relativa_hz"] = df_picos["frecuencia_hz"] - f_dominante
    return df_picos, f_dominante

def frecuencia_dominante_vs_tiempo(y, sr, nperseg, noverlap):
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    idx_dominante = np.argmax(mag, axis=0)
    f_dom_t = f[idx_dominante]
    estadisticas = {
        "f_min": float(np.min(f_dom_t)),
        "f_max": float(np.max(f_dom_t)),
        "f_media": float(np.mean(f_dom_t)),
        "f_std": float(np.std(f_dom_t)),
        "f_rango": float(np.max(f_dom_t) - np.min(f_dom_t)),
    }
    return t, f_dom_t, estadisticas

def envolvente_temporal_amplitud(y):
    """Envolvente TEMPORAL de amplitud, calculada mediante la magnitud de la
    señal analítica (transformada de Hilbert). Se usa para detección de
    pulsos y modulación de amplitud, NUNCA para la FFT (eso es la envolvente
    ESPECTRAL, calculada arriba)."""
    return np.abs(hilbert(y))

def detectar_pulsos(y, sr, prominence, distance_s, height):
    envolvente = envolvente_temporal_amplitud(y)
    distancia_muestras = max(1, int(distance_s * sr))
    picos_idx, _ = find_peaks(envolvente, prominence=prominence, distance=distancia_muestras, height=height)
    if len(picos_idx) < 2:
        return None
    tiempos_pulsos = picos_idx / sr
    intervalos = np.diff(tiempos_pulsos)
    duracion = len(y) / sr
    resultado = {
        "n_pulsos": len(picos_idx),
        "tasa_pulsos_por_s": len(picos_idx) / duracion,
        "intervalo_medio_s": float(np.mean(intervalos)),
        "intervalo_std_s": float(np.std(intervalos)),
        "tiempos_pulsos": tiempos_pulsos,
    }
    return resultado

def calcular_modulacion_frecuencia(f_dom_t, umbral_periodicidad=0.6):
    """Calcula estadísticas de la modulación de frecuencia dominante en el
    tiempo. Solo se etiqueta como 'vibrato' si la autocorrelación de la
    serie muestra un pico periódico claro (por encima de umbral_periodicidad
    tras normalizar); en caso contrario se reporta como 'modulación de
    frecuencia' sin más interpretación."""
    centro = float(np.mean(f_dom_t))
    f_min = float(np.min(f_dom_t))
    f_max = float(np.max(f_dom_t))
    rango = f_max - f_min
    profundidad = rango / centro if centro > 0 else np.nan

    serie = f_dom_t - np.mean(f_dom_t)
    es_periodica = False
    if len(serie) > 4 and np.std(serie) > 0:
        autocorr = np.correlate(serie, serie, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]
        picos_ac, _ = find_peaks(autocorr[1:], height=umbral_periodicidad)
        es_periodica = len(picos_ac) > 0

    etiqueta = "vibrato (oscilación periódica detectada)" if es_periodica else "modulación de frecuencia (sin periodicidad clara)"

    return {
        "frecuencia_central_hz": centro,
        "frecuencia_min_hz": f_min,
        "frecuencia_max_hz": f_max,
        "rango_modulacion_hz": rango,
        "profundidad_modulacion": profundidad,
        "es_periodica": es_periodica,
        "clasificacion": etiqueta,
    }

def calcular_modulacion_amplitud(envolvente, sr):
    amp_media = float(np.mean(envolvente))
    amp_max = float(np.max(envolvente))
    amp_min = float(np.min(envolvente))
    rango = amp_max - amp_min

    env_sin_dc = envolvente - np.mean(envolvente)
    periodicidad_hz = np.nan
    if len(env_sin_dc) > 8:
        X = np.fft.rfft(env_sin_dc)
        freqs = np.fft.rfftfreq(len(env_sin_dc), d=1 / sr)
        mag = np.abs(X)
        if len(mag) > 1:
            idx_max = np.argmax(mag[1:]) + 1  # excluir componente DC
            periodicidad_hz = float(freqs[idx_max])

    return {
        "amplitud_media": amp_media,
        "amplitud_max": amp_max,
        "amplitud_min": amp_min,
        "rango_amplitud": rango,
        "periodicidad_modulacion_hz": periodicidad_hz,
    }

def centroide_y_ancho_banda(freqs, magnitud_db):
    magnitud_lineal = 10 ** (magnitud_db / 20)
    suma_mag = np.sum(magnitud_lineal)
    if suma_mag == 0:
        return np.nan, np.nan
    centroide = np.sum(freqs * magnitud_lineal) / suma_mag
    ancho = np.sqrt(np.sum(((freqs - centroide) ** 2) * magnitud_lineal) / suma_mag)
    return float(centroide), float(ancho)

def graficar_forma_onda(y, sr, titulo, color):
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, y, linewidth=0.6, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    fig.tight_layout()
    return fig

def graficar_forma_onda_comparativa(procesados):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for sesion in SESIONES:
        y = procesados[sesion]["procesado"]
        sr = procesados[sesion]["sr"]
        t = np.arange(len(y)) / sr
        ax.plot(t, y, linewidth=0.5, alpha=0.8, label=ETIQUETAS[sesion], color=colores[sesion])
    ax.set_title("Señal temporal — Mañana vs. Tarde (tiempo relativo al inicio de cada intervalo)")
    ax.set_xlabel("Tiempo (s, relativo al inicio del intervalo)")
    ax.set_ylabel("Amplitud")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_sonograma(f, t, Sxx_db, titulo, f_max):
    fig, ax = plt.subplots(figsize=(10, 4))
    mascara = f <= f_max
    pcm = ax.pcolormesh(t, f[mascara], Sxx_db[mascara, :], shading="gouraud", cmap="magma")
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Intensidad (dB)")
    fig.tight_layout()
    return fig

def graficar_fft(freqs, magnitud_db, envolvente, df_picos, titulo, n_top=8, mostrar_picos=True):
    """Grafica el espectro (FFT) y su envolvente ESPECTRAL. Si
    mostrar_picos=True y hay máximos detectados, los resalta y etiqueta con
    su frecuencia (Hz)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, magnitud_db, linewidth=0.5, color="gray", alpha=0.6, label="Espectro (FFT)")
    ax.plot(freqs, envolvente, linewidth=1.5, color="black", label="Envolvente espectral")
    if mostrar_picos and df_picos is not None and len(df_picos) > 0:
        top = df_picos.head(n_top)
        ax.scatter(top["frecuencia_hz"], top["magnitud_db"], color="red", zorder=5, label="Máximos detectados")
        for _, fila in top.iterrows():
            ax.annotate(f"{fila['frecuencia_hz']:.0f} Hz",
                        (fila["frecuencia_hz"], fila["magnitud_db"]),
                        textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (dB)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_comparacion_fft(resultados):
    fig, ax = plt.subplots(figsize=(10, 4))
    for sesion in SESIONES:
        freqs = resultados[sesion]["espectro"]["freqs"]
        magnitud_db = resultados[sesion]["espectro"]["magnitud_db"]
        ax.plot(freqs, magnitud_db, label=ETIQUETAS[sesion], color=colores[sesion], linewidth=0.8)
    ax.set_xlim(0, max(frecuencia_max_sonograma, frecuencia_max))
    ax.set_title("Comparación FFT — Mañana vs. Tarde")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (dB)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_comparacion_maximos_espectrales(tabla_maximos):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for etiqueta_sesion in ["Mañana", "Tarde"]:
        sub = tabla_maximos[tabla_maximos["sesion"] == etiqueta_sesion]
        color = colores["manana"] if etiqueta_sesion == "Mañana" else colores["tarde"]
        ax.scatter(sub["frecuencia_hz"], sub["magnitud"], alpha=0.4, s=15, label=etiqueta_sesion, color=color)
    ax.set_title("Máximos espectrales detectados (intervalos seleccionados, por ventanas) — Mañana vs. Tarde")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (dB)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_frecuencia_vs_tiempo(t, f_dom_t, titulo, color):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, f_dom_t, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    fig.tight_layout()
    return fig

def graficar_frecuencia_vs_tiempo_comparativa(resultados):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for sesion in SESIONES:
        t = resultados[sesion]["frecuencia_vs_tiempo"]["t"]
        f_dom_t = resultados[sesion]["frecuencia_vs_tiempo"]["f_dom_t"]
        ax.plot(t, f_dom_t, label=ETIQUETAS[sesion], color=colores[sesion])
    ax.set_title("Comparación — Frecuencia dominante vs. tiempo (intervalo seleccionado)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_comparacion_frecuencia_ventanas(tabla_resultados):
    fig, ax = plt.subplots(figsize=(10, 4))
    for etiqueta_sesion in ["Mañana", "Tarde"]:
        sub = tabla_resultados[tabla_resultados["sesion"] == etiqueta_sesion]
        color = colores["manana"] if etiqueta_sesion == "Mañana" else colores["tarde"]
        ax.plot(sub["inicio_s"], sub["frecuencia_dominante"], marker="o", label=etiqueta_sesion, color=color)
    ax.set_title("Frecuencia dominante por ventana (intervalos seleccionados) — Mañana y Tarde")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia dominante (Hz)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_envolvente_pulsos(y, sr, envolvente, resultado_pulsos, titulo):
    """Envolvente TEMPORAL de amplitud (Hilbert) con los pulsos detectados
    marcados sobre ella. No debe confundirse con la envolvente ESPECTRAL de
    la FFT (graficar_fft)."""
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, envolvente, linewidth=0.8, color="black", label="Envolvente temporal")
    if resultado_pulsos is not None:
        tiempos = resultado_pulsos["tiempos_pulsos"]
        idx = np.clip((tiempos * sr).astype(int), 0, len(envolvente) - 1)
        ax.scatter(tiempos, envolvente[idx], color="red", s=18, zorder=5,
                   label=f"Pulsos detectados (n={resultado_pulsos['n_pulsos']})")
    else:
        ax.text(0.5, 0.9, "No se detectaron pulsos confiables con los parámetros actuales",
                transform=ax.transAxes, ha="center", fontsize=9, color="darkred")
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (envolvente)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_comparacion_pulsos(resultados):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, sesion in enumerate(SESIONES, start=1):
        pulsos_sesion = resultados[sesion]["pulsos"]
        if pulsos_sesion is not None:
            ax.eventplot(pulsos_sesion["tiempos_pulsos"], lineoffsets=i, colors=colores[sesion])
    ax.set_yticks(range(1, len(SESIONES) + 1))
    ax.set_yticklabels([ETIQUETAS[s] for s in SESIONES])
    ax.set_title("Tasa de pulsos — comparación temporal (intervalo seleccionado)")
    ax.set_xlabel("Tiempo (s)")
    fig.tight_layout()
    return fig

def graficar_comparacion_rms(tabla_resultados):
    fig, ax = plt.subplots(figsize=(10, 4))
    for etiqueta_sesion in ["Mañana", "Tarde"]:
        sub = tabla_resultados[tabla_resultados["sesion"] == etiqueta_sesion]
        color = colores["manana"] if etiqueta_sesion == "Mañana" else colores["tarde"]
        ax.plot(sub["inicio_s"], sub["rms"], marker="o", label=etiqueta_sesion, color=color)
    ax.set_title("RMS vs. tiempo (intervalos seleccionados, por ventanas) — Mañana y Tarde")
    ax.set_xlabel("Tiempo (s, dentro de cada grabación)")
    ax.set_ylabel("RMS")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_distribucion_frecuencia_dominante(tabla_resultados):
    fig, ax = plt.subplots(figsize=(8, 4))
    for etiqueta_sesion in ["Mañana", "Tarde"]:
        sub = tabla_resultados[tabla_resultados["sesion"] == etiqueta_sesion]
        color = colores["manana"] if etiqueta_sesion == "Mañana" else colores["tarde"]
        ax.hist(sub["frecuencia_dominante"].dropna(), bins=15, alpha=0.6, label=etiqueta_sesion, color=color)
    ax.set_title("Distribución de frecuencias dominantes (intervalos seleccionados)")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Frecuencia de aparición (conteo)")
    ax.legend()
    fig.tight_layout()
    return fig

def graficar_distribucion_rms(tabla_resultados):
    fig, ax = plt.subplots(figsize=(8, 4))
    for etiqueta_sesion in ["Mañana", "Tarde"]:
        sub = tabla_resultados[tabla_resultados["sesion"] == etiqueta_sesion]
        color = colores["manana"] if etiqueta_sesion == "Mañana" else colores["tarde"]
        ax.hist(sub["rms"].dropna(), bins=15, alpha=0.6, label=etiqueta_sesion, color=color)
    ax.set_title("Distribución de RMS (toda la grabación)")
    ax.set_xlabel("RMS")
    ax.set_ylabel("Frecuencia de aparición (conteo)")
    ax.legend()
    fig.tight_layout()
    return fig

def tabla_modulacion_frecuencia_comparativa(resultados):
    filas = []
    for sesion in SESIONES:
        fila = dict(resultados[sesion]["modulacion_frecuencia"])
        fila["sesion"] = ETIQUETAS[sesion]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("sesion")

def tabla_modulacion_amplitud_comparativa(resultados):
    filas = []
    for sesion in SESIONES:
        fila = dict(resultados[sesion]["modulacion_amplitud"])
        fila["sesion"] = ETIQUETAS[sesion]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("sesion")

def analizar_ventana(y_ventana, sr):
    y_sin_dc = remover_dc(y_ventana)
    y_filtrado = filtro_pasabanda(y_sin_dc, sr, frecuencia_min, frecuencia_max, orden_filtro)
    y_proc = reducir_ruido(y_filtrado, sr, aplicar_reduccion_ruido)

    metricas = calcular_metricas_temporales(y_proc, sr)
    freqs, magnitud_db = calcular_fft(y_proc, sr)
    df_picos, f_dom = detectar_maximos_espectrales(freqs, magnitud_db, prominence, distance, height)
    centroide, ancho_banda = centroide_y_ancho_banda(freqs, magnitud_db)

    resultado_pulsos = detectar_pulsos(y_proc, sr, prominence_pulsos, distance_pulsos_s, height_pulsos)
    tasa_pulsos = resultado_pulsos["tasa_pulsos_por_s"] if resultado_pulsos else np.nan

    try:
        nperseg_local = min(nperseg, len(y_proc))
        noverlap_local = min(noverlap, max(0, nperseg_local - 1))
        _, f_dom_t_ventana, _ = frecuencia_dominante_vs_tiempo(
            y_proc, sr, nperseg_local, noverlap_local
        )
        mod_frec = calcular_modulacion_frecuencia(f_dom_t_ventana) if len(f_dom_t_ventana) > 2 else None
    except Exception:
        mod_frec = None

    env = envolvente_temporal_amplitud(y_proc)
    mod_amp = calcular_modulacion_amplitud(env, sr)

    fila = {
        "rms": metricas["rms"],
        "energia": metricas["energia"],
        "amplitud_max": metricas["amplitud_max"],
        "frecuencia_dominante": f_dom if f_dom is not None else np.nan,
        "centroide_espectral": centroide,
        "ancho_banda": ancho_banda,
        "frecuencia_min": mod_frec["frecuencia_min_hz"] if mod_frec else np.nan,
        "frecuencia_max": mod_frec["frecuencia_max_hz"] if mod_frec else np.nan,
        "tasa_pulsos": tasa_pulsos,
        "modulacion_frecuencia": mod_frec["rango_modulacion_hz"] if mod_frec else np.nan,
        "modulacion_amplitud": mod_amp["rango_amplitud"],
    }
    return fila, df_picos

def _boton_descargar_figura(fig, nombre_archivo):
    boton = widgets.Button(description="⬇️ Descargar imagen", layout=widgets.Layout(width="170px"))
    salida = widgets.Output()

    def _descargar(_boton=None):
        with salida:
            salida.clear_output()
            os.makedirs(CARPETA_FIGURAS, exist_ok=True)
            ruta = os.path.join(CARPETA_FIGURAS, nombre_archivo)
            fig.savefig(ruta, dpi=150, bbox_inches="tight")
            print(f"Guardado: {ruta}")
            if EN_COLAB:
                files.download(ruta)

    boton.on_click(_descargar)
    return widgets.VBox([boton, salida])

def _boton_descargar_tabla(df, nombre_csv):
    boton = widgets.Button(description="⬇️ Descargar CSV", layout=widgets.Layout(width="170px"))
    salida = widgets.Output()

    def _descargar(_boton=None):
        with salida:
            salida.clear_output()
            os.makedirs(CARPETA_TABLAS, exist_ok=True)
            ruta = os.path.join(CARPETA_TABLAS, nombre_csv)
            df.to_csv(ruta, index=True)
            print(f"Guardado: {ruta}")
            if EN_COLAB:
                files.download(ruta)

    boton.on_click(_descargar)
    return widgets.VBox([boton, salida])

def agregar_figura(fig, nombre_archivo, contenedor):
    """Muestra una figura ya calculada dentro del 'contenedor' (widget VBox)
    de la interfaz, junto con su botón de descarga individual, y la registra
    para el guardado masivo final. No recalcula nada: recibe la figura ya
    construida a partir de los datos en caché."""
    registrar_figura(fig, nombre_archivo)
    salida_fig = widgets.Output()
    with salida_fig:
        display(fig)
    plt.close(fig)
    bloque = widgets.VBox(
        [salida_fig, _boton_descargar_figura(fig, nombre_archivo)],
        layout=widgets.Layout(border="1px solid #ddd", margin="6px 0px", padding="6px"),
    )
    contenedor.children = contenedor.children + (bloque,)

def agregar_tabla(df, nombre_csv, contenedor, titulo=None):
    """Muestra una tabla ya calculada dentro del 'contenedor' de la interfaz,
    junto con su botón de descarga individual, y la registra para el
    guardado masivo final."""
    registrar_tabla(df, nombre_csv)
    salida_tabla = widgets.Output()
    with salida_tabla:
        if titulo:
            print(titulo)
        display(df)
    bloque = widgets.VBox(
        [salida_tabla, _boton_descargar_tabla(df, nombre_csv)],
        layout=widgets.Layout(border="1px solid #ddd", margin="6px 0px", padding="6px"),
    )
    contenedor.children = contenedor.children + (bloque,)

def _titulo(sesion, base):
    return f"{base} — {ETIQUETAS[sesion]}"

def analizar(_boton=None):
    seleccion = [clave for clave, cb in opciones_checkboxes.items() if cb.value]

    with salida_estado:
        salida_estado.clear_output()
        if not seleccion:
            print("Selecciona al menos una visualización (casilla) antes de pulsar 'Analizar'.")
            return
        print(f"Mostrando: {', '.join(seleccion)} — modo: {selector_sesion.label}")

    if not seleccion:
        return

    sesion = selector_sesion.value

    for clave in seleccion:
        if sesion != "comparar":
            s = sesion
            datos = RESULTADOS[s]

            if clave == "senal_temporal":
                fig = graficar_forma_onda(datos["y_procesado"], datos["sr"],
                                          _titulo(s, "Señal temporal (procesada)"), colores[s])
                agregar_figura(fig, f"{s}_senal.png", contenedor_resultados)

            elif clave == "sonograma":
                fig = graficar_sonograma(datos["sonograma"]["f"], datos["sonograma"]["t"],
                                         datos["sonograma"]["Sxx_db"], _titulo(s, "Sonograma"),
                                         frecuencia_max_sonograma)
                agregar_figura(fig, f"{s}_sonograma.png", contenedor_resultados)

            elif clave == "fft_envolvente":
                fig = graficar_fft(datos["espectro"]["freqs"], datos["espectro"]["magnitud_db"],
                                   datos["espectro"]["envolvente"], None,
                                   _titulo(s, "FFT y envolvente espectral"), mostrar_picos=False)
                agregar_figura(fig, f"{s}_fft_envolvente.png", contenedor_resultados)

            elif clave == "maximos_espectrales":
                df_picos = datos["maximos"]["df_picos"]
                fig = graficar_fft(datos["espectro"]["freqs"], datos["espectro"]["magnitud_db"],
                                   datos["espectro"]["envolvente"], df_picos,
                                   _titulo(s, "Máximos espectrales"),
                                   n_top=n_maximos_reportar, mostrar_picos=True)
                agregar_figura(fig, f"{s}_maximos_espectrales.png", contenedor_resultados)
                if df_picos is not None and len(df_picos) > 0:
                    agregar_tabla(df_picos.head(n_maximos_reportar), f"{s}_tabla_maximos_espectrales.csv",
                                 contenedor_resultados, titulo=f"Máximos espectrales detectados — {ETIQUETAS[s]}")

            elif clave == "envolvente_pulsos":
                fig = graficar_envolvente_pulsos(datos["y_procesado"], datos["sr"],
                                                 datos["envolvente_temporal"], datos["pulsos"],
                                                 _titulo(s, "Envolvente temporal y pulsos"))
                agregar_figura(fig, f"{s}_envolvente_pulsos.png", contenedor_resultados)

            elif clave == "frecuencia_tiempo":
                fig = graficar_frecuencia_vs_tiempo(datos["frecuencia_vs_tiempo"]["t"],
                                                    datos["frecuencia_vs_tiempo"]["f_dom_t"],
                                                    _titulo(s, "Frecuencia dominante vs. tiempo"), colores[s])
                agregar_figura(fig, f"{s}_frecuencia_vs_tiempo.png", contenedor_resultados)

            elif clave == "modulacion_frecuencia":
                df_mod = pd.DataFrame([datos["modulacion_frecuencia"]], index=[ETIQUETAS[s]])
                agregar_tabla(df_mod, f"{s}_modulacion_frecuencia.csv", contenedor_resultados,
                             titulo=f"Modulación de frecuencia — {ETIQUETAS[s]}")

            elif clave == "modulacion_amplitud":
                df_mod = pd.DataFrame([datos["modulacion_amplitud"]], index=[ETIQUETAS[s]])
                agregar_tabla(df_mod, f"{s}_modulacion_amplitud.csv", contenedor_resultados,
                             titulo=f"Modulación de amplitud — {ETIQUETAS[s]}")

            elif clave == "resumen_numerico":
                fila = dict(datos["metricas"])
                fila["frecuencia_dominante_hz"] = datos["maximos"]["f_dominante"]
                fila["centroide_espectral_hz"] = datos["espectro"]["centroide"]
                fila["ancho_banda_hz"] = datos["espectro"]["ancho_banda"]
                fila["tasa_pulsos_por_s"] = datos["pulsos"]["tasa_pulsos_por_s"] if datos["pulsos"] else np.nan
                df_resumen = pd.DataFrame([fila], index=[ETIQUETAS[s]])
                agregar_tabla(df_resumen, f"{s}_resumen_numerico.csv", contenedor_resultados,
                             titulo=f"Resumen numérico del intervalo — {ETIQUETAS[s]}")

        else:
            # ---- Modo "Comparar mañana vs. tarde" ----
            if clave == "senal_temporal":
                fig = graficar_forma_onda_comparativa(procesados)
                agregar_figura(fig, "comparacion_senal.png", contenedor_resultados)

            elif clave == "sonograma":
                for s in SESIONES:
                    datos = RESULTADOS[s]
                    fig = graficar_sonograma(datos["sonograma"]["f"], datos["sonograma"]["t"],
                                             datos["sonograma"]["Sxx_db"], _titulo(s, "Sonograma"),
                                             frecuencia_max_sonograma)
                    agregar_figura(fig, f"{s}_sonograma.png", contenedor_resultados)

            elif clave == "fft_envolvente":
                fig = graficar_comparacion_fft(RESULTADOS)
                agregar_figura(fig, "comparacion_fft.png", contenedor_resultados)

            elif clave == "maximos_espectrales":
                fig = graficar_comparacion_maximos_espectrales(tabla_maximos)
                agregar_figura(fig, "comparacion_maximos_espectrales.png", contenedor_resultados)

            elif clave == "envolvente_pulsos":
                fig = graficar_comparacion_pulsos(RESULTADOS)
                agregar_figura(fig, "comparacion_pulsos.png", contenedor_resultados)

            elif clave == "frecuencia_tiempo":
                fig1 = graficar_frecuencia_vs_tiempo_comparativa(RESULTADOS)
                agregar_figura(fig1, "comparacion_frecuencia.png", contenedor_resultados)
                fig2 = graficar_comparacion_frecuencia_ventanas(tabla_resultados)
                agregar_figura(fig2, "comparacion_frecuencia_ventanas.png", contenedor_resultados)

            elif clave == "modulacion_frecuencia":
                df_mod = tabla_modulacion_frecuencia_comparativa(RESULTADOS)
                agregar_tabla(df_mod, "comparacion_modulacion_frecuencia.csv", contenedor_resultados,
                             titulo="Modulación de frecuencia — Mañana vs. Tarde")

            elif clave == "modulacion_amplitud":
                df_mod = tabla_modulacion_amplitud_comparativa(RESULTADOS)
                agregar_tabla(df_mod, "comparacion_modulacion_amplitud.csv", contenedor_resultados,
                             titulo="Modulación de amplitud — Mañana vs. Tarde")

            elif clave == "resumen_numerico":
                agregar_tabla(tabla_comparacion, "comparacion_estadistica_detallada.csv", contenedor_resultados,
                             titulo="Comparación estadística detallada (Mañana vs. Tarde):")
                agregar_tabla(resumen_comparativo, "resumen_comparativo.csv", contenedor_resultados,
                             titulo="Resumen comparativo (medias):")
                fig_rms = graficar_comparacion_rms(tabla_resultados)
                agregar_figura(fig_rms, "comparacion_rms.png", contenedor_resultados)
                fig_dist_f = graficar_distribucion_frecuencia_dominante(tabla_resultados)
                agregar_figura(fig_dist_f, "distribucion_frecuencia_dominante.png", contenedor_resultados)
                fig_dist_r = graficar_distribucion_rms(tabla_resultados)
                agregar_figura(fig_dist_r, "distribucion_rms.png", contenedor_resultados)

def reproducir(_boton=None):
    with salida_audio:
        salida_audio.clear_output()
        sesion = selector_sesion.value
        sesiones_a_reproducir = SESIONES if sesion == "comparar" else [sesion]
        for s in sesiones_a_reproducir:
            ini, fin = intervalos[s]
            print(f"Reproduciendo segmento — {ETIQUETAS[s]} [{ini}s, {fin}s]")
            display(Audio(segmentos[s]["y"], rate=segmentos[s]["sr"]))

def limpiar(_boton=None):
    contenedor_resultados.children = ()
    with salida_estado:
        salida_estado.clear_output()
        print("Gráficas y tablas eliminadas de la pantalla. Los datos calculados siguen disponibles "
              "en memoria: puedes volver a analizar al instante, sin recalcular nada.")

def guardar_todo(_boton=None):
    with salida_guardado:
        salida_guardado.clear_output()
        print("Guardando resultados en disco...")

        os.makedirs(CARPETA_FIGURAS, exist_ok=True)
        os.makedirs(CARPETA_TABLAS, exist_ok=True)

        for nombre_archivo, fig in FIGURAS_PENDIENTES:
            fig.savefig(os.path.join(CARPETA_FIGURAS, nombre_archivo), dpi=150, bbox_inches="tight")
        print(f"  {len(FIGURAS_PENDIENTES)} figura(s) guardadas en '{CARPETA_FIGURAS}/'.")

        for nombre_csv, df in TABLAS_PENDIENTES.items():
            df.to_csv(os.path.join(CARPETA_RESULTADOS, nombre_csv), index=True)
            df.to_csv(os.path.join(CARPETA_TABLAS, nombre_csv), index=True)
        print(f"  {len(TABLAS_PENDIENTES)} tabla(s) guardadas en '{CARPETA_RESULTADOS}/' y '{CARPETA_TABLAS}/'.")

        nombre_zip = "resultados_chicharras.zip"
        carpeta_zip_raiz = "resultados_chicharras"

        if os.path.exists(carpeta_zip_raiz):
            shutil.rmtree(carpeta_zip_raiz)
        shutil.copytree(CARPETA_RESULTADOS, carpeta_zip_raiz)

        if os.path.exists(nombre_zip):
            os.remove(nombre_zip)

        with zipfile.ZipFile(nombre_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for carpeta_raiz, _, archivos in os.walk(carpeta_zip_raiz):
                for nombre_archivo in archivos:
                    ruta_completa = os.path.join(carpeta_raiz, nombre_archivo)
                    ruta_relativa = os.path.relpath(ruta_completa, ".")
                    zf.write(ruta_completa, ruta_relativa)

        print(f"  ZIP creado: {nombre_zip}")

        if EN_COLAB:
            files.download(nombre_zip)
            print("Descarga iniciada.")
        else:
            print(f"Archivo listo en: {os.path.abspath(nombre_zip)}")

        print("\nListo. Todos los resultados quedaron guardados y descargados.")