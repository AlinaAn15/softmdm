import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from softmdm import SoftMDM

st.title("Мягкий вариант MDM-алгоритма для задачи Сильвестра с анимацией")

st.sidebar.header("Настройка")
c = st.sidebar.number_input("Введите значение штрафного множителя с (>0)",
                            min_value=0.000, value=0.01, format="%.5f")

input_option = st.sidebar.selectbox("Как ввести точки?", ["Ручной ввод", "Сгенерировать случайные"])

if "pts" not in st.session_state:
    st.session_state.pts = None
if "res" not in st.session_state:
    st.session_state.res = None
if "x_min" not in st.session_state:
    st.session_state.x_min = None
if "x_max" not in st.session_state:
    st.session_state.x_max = None
if "y_min" not in st.session_state:
    st.session_state.y_min = None
if "y_max" not in st.session_state:
    st.session_state.y_max = None

if input_option == "Ручной ввод":
    user_input = st.text_area("Введите точки в формате [x1,y1]; [x2,y2]; ...", height=150)
    if st.button("Применить введенные точки"):
        try:
            points = []
            for part in user_input.strip().split(';'):
                coords = part.strip().strip('[]').split(',')
                point = [float(coords[0]), float(coords[1])]
                points.append(point)
            pts_array = np.array(points)
            st.session_state.pts = pts_array

            st.session_state.x_min = np.min(pts_array[:, 0])
            st.session_state.x_max = np.max(pts_array[:, 0])
            st.session_state.y_min = np.min(pts_array[:, 1])
            st.session_state.y_max = np.max(pts_array[:, 1])

            st.success(f"Загружено {len(st.session_state.pts)} точек.")
            st.session_state.res = None
        except Exception as e:
            st.error(f"Не получилось преобразовать: {e}")

elif input_option == "Сгенерировать случайные":
    num = st.sidebar.number_input("Количество точек", min_value=10, max_value=500, value=100, step=10)
    x_min_input = st.sidebar.number_input("Минимум по X", min_value=-100, max_value=100, value=-15, step=1)
    x_max_input = st.sidebar.number_input("Максимум по X", min_value=-100, max_value=100, value=15, step=1)
    y_min_input = st.sidebar.number_input("Минимум по Y", min_value=-100, max_value=100, value=-15, step=1)
    y_max_input = st.sidebar.number_input("Максимум по Y", min_value=-100, max_value=100, value=15, step=1)

    if st.sidebar.button("Сгенерировать"):
        pts_array = np.random.rand(num, 2)
        pts_array[:, 0] = pts_array[:, 0] * (x_max_input - x_min_input) + x_min_input
        pts_array[:, 1] = pts_array[:, 1] * (y_max_input - y_min_input) + y_min_input
        st.session_state.pts = pts_array

        st.session_state.x_min = x_min_input
        st.session_state.x_max = x_max_input
        st.session_state.y_min = y_min_input
        st.session_state.y_max = y_max_input

        st.success(f"Сгенерировано {num} точек")
        st.session_state.res = None

if st.session_state.pts is not None:
    if st.session_state.res is None:
        st.session_state.res = SoftMDM(st.session_state.pts, c)

    res = st.session_state.res

    st.write("### Результат")
    center = res['center']
    radius = res['radius']

    st.write(f"Центр минимального шара : ({center[0]:.3f}, {center[1]:.3f})")
    st.write(f"Радиус : {radius:.3f}")

    history = res['history']

    frame = st.slider("Шаг алгоритма", 0, len(history) - 1, 0)
    st.write(f"Δ на шаге {frame}: {history[frame]['Delta']:.6f}")

    center_step = history[frame]["x"]
    radius_step = history[frame]["radius"]

    distances = np.linalg.norm(st.session_state.pts - center_step, axis=1)
    inner_indices = distances <= radius_step
    outer_indices = distances > radius_step

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axhline(0, color='#000000', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='#000000', linestyle='-', linewidth=0.5)

    ax.scatter(st.session_state.pts[inner_indices, 0], st.session_state.pts[inner_indices, 1],
               color='#1f77b4', edgecolor='#000000', s=30, label='Внутри')
    ax.scatter(st.session_state.pts[outer_indices, 0], st.session_state.pts[outer_indices, 1],
               color='#ff7f0e', edgecolor='#000000', s=30, label='Снаружи')

    circle = plt.Circle((center_step[0], center_step[1]), radius_step,
                        color='#1f77b4', fill=False, lw=2, linestyle='-', alpha=0.9)
    ax.add_artist(circle)

    ax.scatter([center_step[0]], [center_step[1]], color='#d62728', marker='*', s=100, label='Центр')

    ax.axis('equal')

    ax.set_xlim(st.session_state.x_min - 10, st.session_state.x_max + 10)
    ax.set_ylim(st.session_state.y_min - 10, st.session_state.y_max + 10)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(visible=False)

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    st.pyplot(fig)