import streamlit as st
import os
from PIL import Image
import numpy as np
import normalization_core
import pandas as pd
import altair as alt
from scipy.interpolate import PchipInterpolator

import streamlit as st
import os
from PIL import Image
import numpy as np
import normalization_core
import pandas as pd
import altair as alt
from scipy.interpolate import PchipInterpolator

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Macenko workstation", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS INJECTION ---
st.markdown("""
<style>
    /* GLOBAL FONTS & COLORS Override */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* REMOVE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */ /* Keep header visible for Sidebar Toggle */ 
    
    /* COMPACT SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E; /* Darker Sidebar */
        border-right: 1px solid #333;
        padding-top: 0rem;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* COMPACT SLIDERS */
    div[data-testid="stSlider"] {
        margin-bottom: -15px; 
    }
    label[data-testid="stLabel"] {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #888;
        font-weight: 600;
        margin-bottom: 0px;
    }
    
    /* EXPANDERS */
    .streamlit-expanderHeader {
        background-color: #252526;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: bold;
        color: #ddd;
        border: 1px solid #333;
    }
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-top: none;
        padding: 10px;
    }
    
    /* IMAGES */
    img {
        border: 1px solid #444;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #2D2D2D;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #aaa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E;
        color: #00ADB5;
        border-top: 2px solid #00ADB5;
    }

</style>
""", unsafe_allow_html=True)

# --- STATE & PATHS ---
BASE_DIR = os.getcwd()
IMAGE_DIR = os.path.join(BASE_DIR, "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "normalized_images")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

try:
    all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
except FileNotFoundError:
    all_files = []

# --- SIDEBAR: CONTROL PANEL ---
# No emojis, clean uppercase headers
with st.sidebar:
    st.markdown("<h3 style='margin-top:0; color:#00ADB5; border-bottom:1px solid #333; padding-bottom:10px;'>MACENKO<span style='color:#fff'>LAB</span></h3>", unsafe_allow_html=True)

    # 1. REFERENCE
    st.markdown("<div style='margin-top:20px; font-size:0.8rem; color:#666; font-weight:bold; margin-bottom:5px;'>REFERENCE STANDARD</div>", unsafe_allow_html=True)
    
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = all_files[0] if all_files else None

    # Compact Target Selector
    tgt_idx = all_files.index(st.session_state.selected_target) if st.session_state.selected_target in all_files else 0
    st.session_state.selected_target = st.selectbox("Target Image", all_files, index=tgt_idx, label_visibility="collapsed")
    
    # Target Preview (Small)
    if st.session_state.selected_target:
        st.image(os.path.join(IMAGE_DIR, st.session_state.selected_target))
        
    with st.expander("BROWSE TARGETS"):
        cols = st.columns(3)
        for i, f in enumerate(all_files[:9]):
            with cols[i%3]:
                if st.button("SET", key=f"t_{f}", help=f):
                    st.session_state.selected_target = f
                    st.rerun()
                st.image(os.path.join(IMAGE_DIR, f), use_container_width=True)

    st.markdown("<div style='margin-top:30px; font-size:0.8rem; color:#666; font-weight:bold; margin-bottom:5px;'>PARAMETER DECK</div>", unsafe_allow_html=True)
    
    # 2. PARAMETER DECK
    tab_norm, tab_grade, tab_fx = st.tabs(["NORM", "GRADE", "FX"])
    
    with tab_norm:
        st.caption("ALGORITHM PARAMETERS")
        io_val = st.slider("Io (TRANSMISSION)", 150, 255, 255, 1, help="**Intensità Luce Trasmessa**\n\n- **A cosa serve**: Definisce la luminosità dello sfondo (vetrino vuoto).\n- **Uso**: Abbassa se l'immagine è troppo chiara. Alza se è troppo scura.\n- **Effetto**: Sposta l'intera scala di luminosità.")
        alpha = st.slider("ALPHA (PERCENTILE)", 0.0, 5.0, 1.0, 0.1, help="**Robustezza Stima Colore**\n\n- **A cosa serve**: Determina quali pixel usare per calcolare i colori 'puri'.\n- **Uso**: Alza se ci sono artefatti scuri o sporcizia.\n- **Effetto**: Più alto = ignora i pixel più estremi/scuri.")
        beta = st.slider("BETA (OD THRESHOLD)", 0.0, 0.5, 0.15, 0.01, help="**Soglia di Trasparenza**\n\n- **A cosa serve**: Ignora i pixel troppo chiari (sfondo) durante il calcolo.\n- **Uso**: Alza se lo sfondo viene confuso con il tessuto.\n- **Effetto**: Esclude il 'bianco' dall'analisi.")
        st.caption("MATCHING")
        use_lum_match = st.checkbox("Luminosity Match", True, help="**Standardizzazione Luminosità**\n\n- **A cosa serve**: Pareggia l'esposizione della foto sorgente a quella target PRIMA dell'analisi.\n- **Effetto**: Corregge foto scure/sovraesposte rendendole coerenti.")
        use_hist_match = st.checkbox("Histogram Match", True, help="**Matching Istogramma Colore**\n\n- **A cosa serve**: Forza la distribuzione dei colori (H&E) ad essere identica al target.\n- **Effetto**: Garantisce che i viola e i rosa siano esattamente della stessa tonalità del riferimento.")

    with tab_grade:
        st.caption("TONE CURVE (SPLINE)")
        # Dense Sliders
        p0 = st.slider("BLACKS (0)", 0, 255, 0, help="**Punto Neri**\nRegola la luminosità delle zone più scure (es. nuclei densi).")
        p1 = st.slider("SHADOWS (64)", 0, 255, 64, help="**Punto Ombre**\nRegola le zone scure ma non nere.")
        p2 = st.slider("MIDTONES (128)", 0, 255, 128, help="**Punto Medi**\nRegola la luminosità generale del tessuto (citoplasma).")
        p3 = st.slider("HIGHLIGHTS (192)", 0, 255, 192, help="**Punto Luci**\nRegola le zone chiare (es. spazi intercellulari).")
        p4 = st.slider("WHITES (255)", 0, 255, 255, help="**Punto Bianchi**\nRegola il 'bianco puro' (es. vacuoli di steatosi).")
        
        # Plot
        curve_points = [(0, p0), (64, p1), (128, p2), (192, p3), (255, p4)]
        x_chart = np.arange(0, 256, 5)
        poly = PchipInterpolator([p[0] for p in curve_points], [p[1] for p in curve_points])
        y_chart = np.clip(poly(x_chart), 0, 255)
        df_chart = pd.DataFrame({'x': x_chart, 'y': y_chart})
        c_chart = alt.Chart(df_chart).mark_line(color='#00ADB5', strokeWidth=2).encode(
            x=alt.X('x', axis=None), y=alt.Y('y', axis=None)
        ).properties(height=80, width=280) # Very compact chart
        st.altair_chart(c_chart)
        
        st.caption("GLOBAL ADJUSTMENTS")
        contrast = st.slider("CONTRAST", 0.5, 2.0, 1.0, 0.05, help="**Contrasto Lineare**\n\n- **Uso**: >1.0 aumenta la differenza tra chiaro e scuro. <1.0 la riduce (più piatto).")
        gamma = st.slider("GAMMA", 0.1, 3.0, 1.0, 0.05, help="**Correzione Gamma**\n\n- **Uso**: <1.0 scurisce i medi. >1.0 schiarisce i medi senza bruciare i bianchi.")
        to_bw = st.checkbox("Monochrome Mode", False, help="Converte l'output finale in Bianco e Nero.")

    with tab_fx:
        st.caption("STEATOSIS ENHANCEMENT")
        clarify_strength = st.slider("CLARIFIER STRENGTH", 0.0, 2.0, 1.0, 0.1, help="**Filtro Pulizia Steatosi**\n\n- **A cosa serve**: Rende i vacuoli (buchi) bianco puro, rimuovendo aloni rosa/grigi.\n- **Uso**: Aumenta finché i buchi non sono netti e puliti.")
        st.caption("DETAIL ENHANCEMENT")
        sharpness = st.slider("UNSHARP MASK", 0.0, 3.0, 1.0, 0.1, help="**Nitidezza (Unsharp Mask)**\n\n- **A cosa serve**: Aumenta il micro-contrasto sui bordi.\n- **Uso**: Valori bassi (0.5-1.0) per definire meglio i nuclei.")

    st.markdown("---")
    if st.button("PROCESS BATCH", type="primary", use_container_width=True):
         st.session_state.run_batch_trigger = True


# --- MAIN VIEWPORT ---
# Top Bar
if 'selected_sample' not in st.session_state:
     st.session_state.selected_sample = all_files[1] if len(all_files)>1 else all_files[0]

# Custom Header Bar
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; background:#252526; padding:10px; border-radius:4px; margin-bottom:10px; border:1px solid #333;">
    <div style="color:#aaa; font-size:0.9rem;"> <span style="color:#00ADB5; font-weight:bold;">ACTIVE SAMPLE:</span> {st.session_state.selected_sample} </div>
    <div style="color:#666; font-size:0.8rem;"> {len(all_files)} FILES LOADED </div>
</div>
""", unsafe_allow_html=True)

# Image Logic
if st.session_state.selected_target and st.session_state.selected_sample:
    with st.spinner("COMPUTING..."):
        t_path = os.path.join(IMAGE_DIR, st.session_state.selected_target)
        s_path = os.path.join(IMAGE_DIR, st.session_state.selected_sample)
        
        t_img = np.array(Image.open(t_path).convert('RGB'))
        s_img = np.array(Image.open(s_path).convert('RGB'))
        
        res_img = normalization_core.process_image(
            t_img, s_img,
            use_lum_match, use_hist_match,
            clarify_strength, sharpness, contrast, gamma,
            curve_points, to_bw, alpha, beta, io_val
        )

    # Viewport
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='text-align:center; color:#666; font-size:0.8rem; margin-bottom:5px;'>ORIGINAL INPUT</div>", unsafe_allow_html=True)
        st.image(s_img, use_container_width=True)
    with col2:
        st.markdown("<div style='text-align:center; color:#00ADB5; font-size:0.8rem; margin-bottom:5px; font-weight:bold;'>NORMALIZED OUTPUT</div>", unsafe_allow_html=True)
        st.image(res_img, use_container_width=True)

    # --- ANALYTICS DASHBOARD ---
    with st.expander("📊 ANALYTICS DASHBOARD", expanded=True):
        st.caption("COLOR DISTRIBUTION METRICS")
        
        def calculate_hist_data(img_np):
            # Calculate histograms per channel
            r_hist, _ = np.histogram(img_np[:,:,0], bins=256, range=(0,256), density=True)
            g_hist, _ = np.histogram(img_np[:,:,1], bins=256, range=(0,256), density=True)
            b_hist, _ = np.histogram(img_np[:,:,2], bins=256, range=(0,256), density=True)
            x = np.arange(256)
            
            # Melt for Altair
            df_r = pd.DataFrame({'val': x, 'density': r_hist, 'channel': 'Red'})
            df_g = pd.DataFrame({'val': x, 'density': g_hist, 'channel': 'Green'})
            df_b = pd.DataFrame({'val': x, 'density': b_hist, 'channel': 'Blue'})
            return pd.concat([df_r, df_g, df_b])

        # Calc
        df_src = calculate_hist_data(s_img)
        df_res = calculate_hist_data(np.array(res_img))
        
        # Chart Helper
        def plot_hist(df, title):
            c = alt.Chart(df).mark_line(strokeWidth=2).encode(
                x=alt.X('val', title="Brightness (0-255)"),
                y=alt.Y('density', title=None, axis=None),
                color=alt.Color('channel', scale=alt.Scale(domain=['Red', 'Green', 'Blue'], range=['#ff4b4b', '#009900', '#4b4bff']), legend=None),
                tooltip=['channel', 'val', 'density']
            ).properties(height=150, title=title)
            return c

        c_a, c_b = st.columns(2)
        with c_a:
            st.altair_chart(plot_hist(df_src, "INPUT SPECTRUM (RGB)"), use_container_width=True)
        with c_b:
            st.altair_chart(plot_hist(df_res, "OUTPUT SPECTRUM (RGB)"), use_container_width=True)
            
        st.info("ℹ️ **INTERPRETAZIONE**: I picchi indicano i colori dominanti. Nel **Target**, il rosa (Eosina) e il viola (Ematossilina) creano curve distinte. L'Output dovrebbe assomigliare alla distribuzione del Target.")

# Footer: Tray
st.markdown("<div style='margin-top:20px; border-top:1px solid #333;'></div>", unsafe_allow_html=True)
with st.expander("IMAGE TRAY", expanded=True):
    # Sliding Window 
    curr_idx = all_files.index(st.session_state.selected_sample)
    start = max(0, curr_idx - 6)
    end = min(len(all_files), start + 12)
    
    cols = st.columns(12)
    for i, idx in enumerate(range(start, end)):
        f = all_files[idx]
        with cols[i]:
            # Highlight active
            border_color = "#00ADB5" if f == st.session_state.selected_sample else "#444"
            st.markdown(f"<div style='border:2px solid {border_color}; border-radius:4px; padding:2px;'>", unsafe_allow_html=True)
            st.image(os.path.join(IMAGE_DIR, f), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("LOAD", key=f"tray_{f}", help=f):
                st.session_state.selected_sample = f
                st.rerun()

# Batch Runner
if st.session_state.get('run_batch_trigger', False):
    st.info("BATCH JOB STARTED...")
    pbar = st.progress(0)
    
    for i, f in enumerate(all_files):
        try:
             src = np.array(Image.open(os.path.join(IMAGE_DIR, f)).convert('RGB'))
             out = normalization_core.process_image(
                t_img, src, # target is static from current
                use_lum_match, use_hist_match,
                clarify_strength, sharpness, contrast, gamma,
                curve_points, to_bw, alpha, beta, io_val
             )
             out.save(os.path.join(OUTPUT_DIR, f))
        except: pass
        pbar.progress((i+1)/len(all_files))
        
    st.success("BATCH COMPLETE")
    st.session_state.run_batch_trigger = False

