# Métodos numéricos en ingeniería mecánica

Proyecto de Análisis Numérico. App interactiva en Streamlit con dos aplicaciones:

1. **Péndulo simple y doble:** integración de EDOs con **Euler** y **Runge-Kutta 4**
   (comparación de trayectorias, conservación de energía y orden de convergencia).
2. **Mecanismo de cuatro barras:** análisis de posición con **Newton-Raphson**
   (iteraciones, convergencia cuadrática, trayectoria del acoplador y sensibilidad al estimado inicial).

Los métodos están implementados desde cero con NumPy en `metodos.py`.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar gratis (Streamlit Community Cloud)

1. Sube estos archivos a un repositorio de GitHub.
2. Entra a https://share.streamlit.io e inicia sesión con GitHub.
3. Elige "New app", selecciona el repositorio y la rama, y como archivo principal `app.py`.
4. Pulsa "Deploy" y comparte el enlace.

## Estructura

```
├── app.py            # interfaz de Streamlit
├── metodos.py        # Euler, RK4, Newton-Raphson y modelos mecánicos
├── requirements.txt  # dependencias
├── README.md
└── .gitignore
```
