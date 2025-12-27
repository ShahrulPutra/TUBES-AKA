import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import sys

st.set_page_config(
    page_title="Analisis Flood Fill: Full White Grid (Nano Seconds)",
    layout="wide"
)

sys.setrecursionlimit(200000)

def flood_fill_recursive(grid, x, y, new_color, old_color):
    rows, cols = grid.shape
    
    if (x < 0) or (x >= rows) or (y < 0) or (y >= cols) or grid[x, y] != old_color:
        return
        

    grid[x, y] = new_color

    flood_fill_recursive(grid, x + 1, y, new_color, old_color) 
    flood_fill_recursive(grid, x - 1, y, new_color, old_color)
    flood_fill_recursive(grid, x, y + 1, new_color, old_color)
    flood_fill_recursive(grid, x, y - 1, new_color, old_color) 

def flood_fill_iterative(grid, start_x, start_y, new_color, old_color):
    rows, cols = grid.shape
    stack = []
    stack.append((start_x, start_y))

    for item in stack:
        x, y = item

        if (x < 0) or (x >= rows) or (y < 0) or (y >= cols) or grid[x, y] != old_color:
            continue
        else:
            grid[x, y] = new_color

            stack.append((x + 1, y))
            stack.append((x - 1, y)) 
            stack.append((x, y + 1)) 
            stack.append((x, y - 1))

def create_full_white_grid(size):
    return np.zeros((size, size), dtype=int)

def plot_grid(grid, title):
    cmap = colors.ListedColormap(['white', 'black', '#FF4B4B'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='#CCCCCC', linewidth=0.5)
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    return fig

st.title("Analisis Flood Fill: Nanosecond Precision ⏱️")
st.markdown("""
Simulasi ini menggunakan satuan waktu **Nanosecond (ns)** untuk presisi tinggi.
$$1 \\text{ ms} = 1.000.000 \\text{ ns}$$
""")

st.sidebar.header("Pengaturan")
grid_size = st.sidebar.slider("Ukuran Grid (N x N)", min_value=10, max_value=200, value=50, step=10)
btn_reset = st.sidebar.button("Reset Grid")

if 'grid_base' not in st.session_state or btn_reset:
    st.session_state.grid_base = create_full_white_grid(grid_size)
    st.session_state.time_rec = None
    st.session_state.time_iter = None
    st.session_state.res_rec = None
    st.session_state.res_iter = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Canvas Awal")
    fig_base = plot_grid(st.session_state.grid_base, "Grid Kosong (0)")
    st.pyplot(fig_base)

with col2:
    st.subheader("Eksekusi Algoritma")
    
    grid_rec = st.session_state.grid_base.copy()
    grid_iter = st.session_state.grid_base.copy()
    
    start_x, start_y = 0, 0
    fill_color = 2 
    target_color = 0 

    c1, c2 = st.columns(2)
    
   
    with c1:
        if st.button("Jalankan Rekursif"):
            try:
               
                start_time = time.perf_counter_ns()
                flood_fill_recursive(grid_rec, start_x, start_y, fill_color, target_color)
                end_time = time.perf_counter_ns()
                
                
                st.session_state.time_rec = end_time - start_time
                st.session_state.res_rec = grid_rec
                
                
                st.success(f"Rekursif: {st.session_state.time_rec:,} ns")
            except RecursionError:
                st.error("Stack Overflow! (Recursion Limit Reached)")
                st.session_state.time_rec = None
                st.session_state.res_rec = None


    with c2:
        if st.button("Jalankan Iteratif"):
            start_time = time.perf_counter_ns()
            flood_fill_iterative(grid_iter, start_x, start_y, fill_color, target_color)
            end_time = time.perf_counter_ns()
            
            st.session_state.time_iter = end_time - start_time
            st.session_state.res_iter = grid_iter
            st.success(f"Iteratif: {st.session_state.time_iter:,} ns")


st.divider()

if st.session_state.res_rec is not None or st.session_state.res_iter is not None:
    r1, r2 = st.columns(2)
    with r1:
        if st.session_state.res_rec is not None:
            st.pyplot(plot_grid(st.session_state.res_rec, "Hasil Rekursif"))
    with r2:
        if st.session_state.res_iter is not None:
            st.pyplot(plot_grid(st.session_state.res_iter, "Hasil Iteratif"))

st.divider()
st.header("📈 Benchmark Nanosecond (Log Scale)")

if st.button("Mulai Benchmark (10x10 s.d 100x100)"):
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    times_rec = []
    times_iter = []
    pixels = []

    prog_bar = st.progress(0)
    
    for i, n in enumerate(sizes):
        prog_bar.progress((i + 1) / len(sizes))
        
        test_grid = np.zeros((n, n), dtype=int)
        

        g_rec = test_grid.copy()
        s = time.perf_counter_ns()
        try:
            flood_fill_recursive(g_rec, 0, 0, 2, 0)
            times_rec.append(time.perf_counter_ns() - s)
        except RecursionError:
            times_rec.append(None)
            
        g_iter = test_grid.copy()
        s = time.perf_counter_ns()
        flood_fill_iterative(g_iter, 0, 0, 2, 0)
        times_iter.append(time.perf_counter_ns() - s)
        
        pixels.append(n * n)


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pixels, times_iter, 'go-', label='Iteratif (Stack)', linewidth=2)
    
    valid_rec = [(p, t) for p, t in zip(pixels, times_rec) if t is not None]
    if valid_rec:
        px, tx = zip(*valid_rec)
        ax.plot(px, tx, 'bx--', label='Rekursif (DFS)')
    
    ax.set_xlabel('Total Piksel (Log Scale)')
    ax.set_ylabel('Waktu (Nanoseconds) (Log Scale)')
    ax.set_title('Perbandingan Waktu Eksekusi (ns)')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
