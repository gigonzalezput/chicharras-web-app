# Aplicación web: Análisis acústico de chicharras

Esta versión transforma el programa de Google Colab `analisis_chicharras_colab.py`
en una aplicación web con Streamlit.

## Ejecutar localmente

1. Instalar Python 3.10 o superior.
2. Abrir una terminal en esta carpeta.
3. Ejecutar:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. El navegador abrirá la aplicación.

## Publicar con un enlace

El proyecto está preparado para servicios que ejecutan aplicaciones Streamlit.
Al subir esta carpeta a un repositorio GitHub y desplegar `app.py`, el servicio
proporcionará una URL pública.

## Diferencias respecto a Colab

- Ya no depende de `ipywidgets`, `IPython.display` ni `google.colab.files`.
- Los dos audios se cargan desde la página web.
- Los parámetros principales son editables desde la barra lateral.
- Se mantiene el análisis de mañana/tarde, FFT, sonograma, máximos espectrales,
  pulsos, modulación, tablas y comparación.
- Las figuras se pueden descargar individualmente y también se puede preparar
  un ZIP con resultados.

## Nota

La aplicación ejecuta Python en el servidor. El computador del usuario solo
necesita un navegador web; no necesita instalar Python.
