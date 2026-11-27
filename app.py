import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from metodos import (
    GRAVEDAD,
    barrido_cuatro_barras,
    cuatro_barras_analitica,
    energia_doble,
    energia_simple,
    integrar,
    lazo_cuatro_barras,
    newton_raphson,
    pendulo_doble,
    pendulo_simple,
)

st.set_page_config(
    page_title="Métodos numéricos en ingeniería mecánica",
    page_icon="⚙️",
    layout="wide",
)
st.title("⚙️ Métodos numéricos en ingeniería mecánica")
st.caption(
    "Proyecto de Análisis Numérico · EDOs (Euler vs RK4) y "
    "sistemas no lineales (Newton-Raphson)"
)

tab1, tab2 = st.tabs(
    ["🕰️ Péndulo (EDOs: Euler vs RK4)", "🔧 Cuatro barras (Newton-Raphson)"]
)


def mostrar(placeholder, fig):
    placeholder.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# PESTAÑA 1: PÉNDULO
# ===========================================================================
@st.cache_data(show_spinner=False)
def simular(modelo, m1, m2, L1, L2, y0, tf, h, metodo):
    f = pendulo_simple(L1) if modelo == "Simple" else pendulo_doble(m1, m2, L1, L2)
    with np.errstate(all="ignore"):
        return integrar(f, np.array(y0, float), tf, h, metodo)


@st.cache_data(show_spinner="Calculando estudio de convergencia…")
def estudio_convergencia(modelo, m1, m2, L1, L2, y0, tf, hs):
    """Error en t = tf contra una solución de referencia (RK4 con h = 0.001)."""
    _, Yref = simular(modelo, m1, m2, L1, L2, y0, tf, 0.001, "rk4")
    ref = Yref[-1]
    errores = {"Euler": [], "RK4": []}
    for h in hs:
        for nombre in errores:
            _, Y = simular(modelo, m1, m2, L1, L2, y0, tf, h, nombre.lower())
            with np.errstate(all="ignore"):
                errores[nombre].append(float(np.linalg.norm(Y[-1] - ref)))
    return errores


def energia(modelo, Y, m1, m2, L1, L2):
    with np.errstate(all="ignore"):
        if modelo == "Simple":
            return energia_simple(Y, m1, L1)
        return energia_doble(Y, m1, m2, L1, L2)


def posiciones(modelo, Y, L1, L2):
    th1 = Y[:, 0]
    x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
    if modelo == "Simple":
        return x1, y1, None, None
    th2 = Y[:, 2]
    return x1, y1, x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)


def orden_observado(hs, errs):
    h = np.array(hs[-3:])
    e = np.array(errs[-3:])
    if not np.all(np.isfinite(e)) or np.any(e <= 0):
        return float("nan")
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def fig_pendulo(modelo, Ys, k, tk, L1, L2):
    fig, ax = plt.subplots(figsize=(5, 5))
    estilos = {"RK4": ("tab:blue", 1.0), "Euler": ("tab:red", 0.6)}
    for nombre, Y in Ys.items():
        color, alpha = estilos[nombre]
        x1, y1, x2, y2 = posiciones(modelo, Y, L1, L2)
        xe, ye = (x1, y1) if x2 is None else (x2, y2)
        ax.plot(xe[: k + 1], ye[: k + 1], color=color, alpha=0.3, lw=1)
        cx, cy = [0, x1[k]], [0, y1[k]]
        if x2 is not None:
            cx.append(x2[k])
            cy.append(y2[k])
        ax.plot(cx, cy, "o-", color=color, alpha=alpha, lw=2, label=nombre)
    R = (L1 + (0 if modelo == "Simple" else L2)) * 1.15
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"t = {tk:.2f} s")
    return fig


with tab1:
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Parámetros")
        modelo = st.radio("Modelo", ["Simple", "Doble"], horizontal=True)
        L1 = st.slider("Longitud L₁ (m)", 0.2, 3.0, 1.0, 0.1)
        m1 = st.slider("Masa m₁ (kg)", 0.1, 5.0, 1.0, 0.1)
        if modelo == "Doble":
            L2 = st.slider("Longitud L₂ (m)", 0.2, 3.0, 1.0, 0.1)
            m2 = st.slider("Masa m₂ (kg)", 0.1, 5.0, 1.0, 0.1)
        else:
            L2, m2 = 1.0, 1.0
        th1_0 = st.slider("Ángulo inicial θ₁ (°)", -170, 170, 90)
        if modelo == "Doble":
            th2_0 = st.slider("Ángulo inicial θ₂ (°)", -170, 170, 0)
            y0 = (np.radians(th1_0), 0.0, np.radians(th2_0), 0.0)
        else:
            y0 = (np.radians(th1_0), 0.0)
        tf = float(st.slider("Tiempo total (s)", 2, 20, 10))
        h = st.select_slider("Paso h (s)", options=[0.1, 0.05, 0.02, 0.01, 0.005], value=0.05)

    Ys = {}
    for nombre in ("RK4", "Euler"):
        t, Ys[nombre] = simular(modelo, m1, m2, L1, L2, y0, tf, h, nombre.lower())
    n = len(t) - 1

    with c2:
        st.subheader("Simulación")
        t_sel = st.slider("Instante t (s)", 0.0, float(t[-1]), 0.0, step=float(h))
        k = min(n, int(round(t_sel / h)))
        placeholder = st.empty()
        mostrar(placeholder, fig_pendulo(modelo, Ys, k, t[k], L1, L2))
        if st.button("▶ Reproducir animación"):
            paso = max(1, n // 100)
            for kk in range(0, n + 1, paso):
                mostrar(placeholder, fig_pendulo(modelo, Ys, kk, t[kk], L1, L2))
                time.sleep(0.02)

    st.subheader("Comparación de métodos")
    escala = (m1 + (m2 if modelo == "Doble" else 0)) * GRAVEDAD * (
        L1 + (L2 if modelo == "Doble" else 0)
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for nombre, Y in Ys.items():
        axs[0].plot(t, np.degrees(Y[:, 0]), label=nombre)
        E = energia(modelo, Y, m1, m2, L1, L2)
        axs[1].semilogy(t, np.maximum(np.abs(E - E[0]) / escala, 1e-16), label=nombre)
    axs[0].set(xlabel="t (s)", ylabel="θ₁ (°)", title="Ángulo θ₁(t)")
    axs[1].set(xlabel="t (s)", ylabel="|E − E₀| / E_ref", title="Error en la energía (se debe conservar)")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Sin fricción la energía mecánica es constante: cualquier variación es error "
        "numérico. Euler la aumenta con el tiempo; RK4 la mantiene casi constante."
    )

    st.subheader("Orden de convergencia")
    hs = (0.1, 0.05, 0.02, 0.01, 0.005)
    tf_c = float(max(1, min(round(tf), 5 if modelo == "Doble" else 10)))
    err = estudio_convergencia(modelo, m1, m2, L1, L2, y0, tf_c, hs)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for nombre, e in err.items():
        ax.loglog(hs, e, "o-", label=nombre)
    h_ref = np.array([hs[-1], hs[0]])
    mid = err["RK4"][-1] if np.isfinite(err["RK4"][-1]) and err["RK4"][-1] > 0 else 1e-6
    ax.loglog(h_ref, mid * (h_ref / hs[-1]) ** 4, "k--", alpha=0.4, label="pendiente 4")
    ax.loglog(h_ref, err["Euler"][-1] * (h_ref / hs[-1]) ** 1, "k:", alpha=0.4, label="pendiente 1")
    ax.set(xlabel="paso h (s)", ylabel="error en t = %.0f s" % tf_c)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.pyplot(fig)
        plt.close(fig)
    with cc2:
        st.metric("Orden observado · Euler", f"{orden_observado(hs, err['Euler']):.2f}")
        st.metric("Orden observado · RK4", f"{orden_observado(hs, err['RK4']):.2f}")
        st.caption(
            "Teórico: Euler = 1, RK4 = 4. En el péndulo doble (caótico) el error "
            "crece muy rápido y el orden observado puede desviarse."
        )

    with st.expander("📐 Fundamento matemático"):
        st.latex(r"\ddot\theta = -\frac{g}{L}\sin\theta \;\Rightarrow\; "
                 r"\mathbf{y}' = \mathbf{f}(t,\mathbf{y}),\quad \mathbf{y}=[\theta,\omega]")
        st.latex(r"\text{Euler: } \mathbf{y}_{n+1}=\mathbf{y}_n+h\,\mathbf{f}(t_n,\mathbf{y}_n)")
        st.latex(r"\text{RK4: } \mathbf{y}_{n+1}=\mathbf{y}_n+\tfrac{h}{6}(k_1+2k_2+2k_3+k_4)")
        st.markdown(
            "El error global de Euler es $O(h)$ y el de RK4 es $O(h^4)$. "
            "El péndulo doble usa las ecuaciones de Lagrange estándar."
        )


# ===========================================================================
# PESTAÑA 2: MECANISMO DE CUATRO BARRAS
# ===========================================================================
@st.cache_data(show_spinner=False)
def barrido(r1, r2, r3, r4, signo, n=181):
    th2 = np.radians(np.linspace(0, 360, n))
    return th2, barrido_cuatro_barras(th2, r1, r2, r3, r4, signo)


def geometria(t2, t3, r2, r3, p, alpha):
    A = r2 * np.array([np.cos(t2), np.sin(t2)])
    B = A + r3 * np.array([np.cos(t3), np.sin(t3)])
    P = A + p * np.array([np.cos(t3 + alpha), np.sin(t3 + alpha)])
    return A, B, P


def fig_mecanismo(r1, r2, r3, p, alpha, t2, t3, lims, tray=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    A, B, P = geometria(t2, t3, r2, r3, p, alpha)
    O4 = np.array([r1, 0.0])
    if tray is not None:
        ax.plot(tray[0], tray[1], color="tab:green", lw=1, label="Trayectoria de P")
    ax.plot([0, r1], [0, 0], "k--", lw=1)
    ax.plot([0, A[0]], [0, A[1]], "o-", color="tab:blue", lw=3, label="Manivela")
    ax.plot([A[0], B[0]], [A[1], B[1]], "o-", color="tab:orange", lw=3, label="Acoplador")
    ax.plot([O4[0], B[0]], [O4[1], B[1]], "o-", color="tab:red", lw=3, label="Balancín")
    ax.fill([A[0], B[0], P[0]], [A[1], B[1], P[1]], color="tab:orange", alpha=0.2)
    ax.plot(P[0], P[1], "s", color="tab:green")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return fig


with tab2:
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Parámetros")
        r1 = st.slider("Bancada r₁", 1.0, 10.0, 4.0, 0.1)
        r2 = st.slider("Manivela r₂", 0.5, 5.0, 1.5, 0.1)
        r3 = st.slider("Acoplador r₃", 1.0, 10.0, 4.0, 0.1)
        r4 = st.slider("Balancín r₄", 1.0, 10.0, 3.5, 0.1)
        p = st.slider("Punto P: distancia desde A", 0.0, 10.0, 2.0, 0.1)
        alpha = np.radians(st.slider("Punto P: ángulo sobre el acoplador (°)", -90, 90, 30))
        signo = 1 if st.radio("Configuración", ["A", "B"], horizontal=True) == "A" else -1
        t2_deg = st.slider("Ángulo de entrada θ₂ (°)", 0, 360, 60)
        tol = st.select_slider("Tolerancia de Newton", options=[1e-2, 1e-4, 1e-8, 1e-12], value=1e-8,
                               format_func=lambda v: f"{v:.0e}")
        manual = st.checkbox("Elegir estimado inicial manualmente")
        if manual:
            g3 = st.slider("θ₃ inicial (°)", -180, 360, 0)
            g4 = st.slider("θ₄ inicial (°)", -180, 360, 0)

        L = sorted([r1, r2, r3, r4])
        if L[0] + L[3] <= L[1] + L[2]:
            st.info("✅ Cumple Grashof: al menos una barra puede girar completa.")
        else:
            st.warning("⚠️ No cumple Grashof: ninguna barra gira completa (solo oscila).")

    th2s, sol = barrido(r1, r2, r3, r4, signo)
    ok = ~np.isnan(sol[:, 0])
    if ok.any():
        A_s, B_s, P_s = geometria(th2s[ok], sol[ok, 0], r2, r3, p, alpha)
        pts = np.hstack([A_s, B_s, P_s, np.array([[0, r1], [0, 0]])])
        mn, mx = pts.min(axis=1) - 0.7, pts.max(axis=1) + 0.7
        lims = (mn[0], mx[0], mn[1], mx[1])
        tray = P_s
    else:
        lims = (-r2 - r3, r1 + r4, -r2 - r3, r2 + r3)
        tray = None

    t2 = np.radians(t2_deg)
    xa = cuatro_barras_analitica(t2, r1, r2, r3, r4, signo)

    with c2:
        st.subheader("Solución con Newton-Raphson")
        if xa is None:
            st.error("El mecanismo no se puede ensamblar con estas longitudes para este θ₂. "
                     "Cambia θ₂ o las dimensiones.")
        else:
            x0 = np.radians([g3, g4]) if manual else xa
            F, J = lazo_cuatro_barras(t2, r1, r2, r3, r4)
            x, hist, conv = newton_raphson(F, J, x0, tol=tol)

            if conv:
                st.success(f"Convergió en {len(hist) - 1} iteraciones.")
            else:
                st.warning("No convergió con ese estimado inicial (se muestra la solución analítica).")
            x_dib = x if conv else xa

            m1_, m2_, m3_ = st.columns(3)
            m1_.metric("θ₃ (°)", f"{np.degrees(x_dib[0]) % 360:.4f}")
            m2_.metric("θ₄ (°)", f"{np.degrees(x_dib[1]) % 360:.4f}")
            m3_.metric("‖F‖ final", f"{hist[-1][2]:.2e}")

            dif = np.degrees(np.abs(np.angle(np.exp(1j * (x_dib - xa)))))
            msg = f"Diferencia vs solución analítica: Δθ₃ = {dif[0]:.1e}°, Δθ₄ = {dif[1]:.1e}°."
            if conv and dif.max() > 1e-3:
                msg += " ⚠️ Newton convergió a otra configuración del mecanismo."
            st.caption(msg)

            placeholder2 = st.empty()
            mostrar(placeholder2, fig_mecanismo(r1, r2, r3, p, alpha, t2, x_dib[0], lims, tray))
            if st.button("▶ Reproducir giro de la manivela") and ok.any():
                for i in range(0, len(th2s), 2):
                    if ok[i]:
                        mostrar(placeholder2, fig_mecanismo(r1, r2, r3, p, alpha, th2s[i],
                                                            sol[i, 0], lims, tray))
                        time.sleep(0.02)

            st.markdown("**Iteraciones de Newton-Raphson**")
            tabla = pd.DataFrame(
                [
                    {
                        "Iteración": k_,
                        "θ₃ (°)": round(float(np.degrees(xx[0])), 6),
                        "θ₄ (°)": round(float(np.degrees(xx[1])), 6),
                        "‖F‖": f"{res:.3e}",
                    }
                    for k_, xx, res in hist
                ]
            )
            st.dataframe(tabla, hide_index=True)

    if xa is not None:
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.semilogy([h_[0] for h_ in hist], [max(h_[2], 1e-17) for h_ in hist], "o-")
            ax.axhline(tol, color="k", ls="--", alpha=0.4, label="tolerancia")
            ax.set(xlabel="iteración", ylabel="‖F(x)‖", title="Convergencia (cuadrática)")
            ax.grid(alpha=0.3, which="both")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        with g2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(np.degrees(th2s), np.degrees(sol[:, 0]), label="θ₃ (acoplador)")
            ax.plot(np.degrees(th2s), np.degrees(sol[:, 1]), label="θ₄ (balancín)")
            ax.axvline(t2_deg, color="k", ls=":", alpha=0.5)
            ax.set(xlabel="θ₂ (°)", ylabel="ángulo (°)", title="Análisis de posición completo")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("📐 Fundamento matemático"):
        st.markdown("Ecuación de cierre del lazo vectorial, con incógnitas $\\theta_3$ y $\\theta_4$:")
        st.latex(r"\mathbf{F}(\theta_3,\theta_4)=\begin{bmatrix}"
                 r"r_2\cos\theta_2+r_3\cos\theta_3-r_4\cos\theta_4-r_1\\"
                 r"r_2\sin\theta_2+r_3\sin\theta_3-r_4\sin\theta_4\end{bmatrix}=\mathbf{0}")
        st.latex(r"\mathbf{J}\,\Delta\mathbf{x}=-\mathbf{F}(\mathbf{x}_k),\qquad "
                 r"\mathbf{x}_{k+1}=\mathbf{x}_k+\Delta\mathbf{x}")
        st.markdown(
            "Para el barrido completo se usa **continuación**: la solución de un ángulo es "
            "el estimado inicial del siguiente. Con un mal estimado, Newton puede converger "
            "a la otra configuración o no converger; pruébalo con la opción manual."
        )
