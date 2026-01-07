import streamlit as st
import os
from PIL import Image
import numpy as np
import normalization_core
import pandas as pd
import altair as alt
from scipy.interpolate import PchipInterpolator
import base64
from streamlit_clickable_images import clickable_images
import json
from PIL.PngImagePlugin import PngInfo

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

# --- HELPERS ---
@st.cache_data
def get_thumbnails(file_list, folder, size=(120, 120)):
    images = []
    for f in file_list:
        path = os.path.join(folder, f)
        with Image.open(path) as img:
            img.thumbnail(size) # Dynamic size
            import io
            with io.BytesIO() as buffer:
                img.save(buffer, format="JPEG", quality=80) 
                img_str = base64.b64encode(buffer.getvalue()).decode()
                images.append(f"data:image/jpeg;base64,{img_str}")
    return images

# Pre-load thumbnails (limit 50 for safety)
with st.spinner("Initializing assets..."):
    all_thumbnails = get_thumbnails(all_files[:50], IMAGE_DIR)

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
        # Grid Layout (3x3 approx)
        tgt_idx_grid = all_files.index(st.session_state.selected_target) if st.session_state.selected_target in all_files else -1
        
        # We reuse all_thumbnails which matches all_files[:50]
        # Limitation: Target selector only shows first 50. Acceptable for now.
        
        clicked_tgt = clickable_images(
            all_thumbnails, 
            titles=all_files[:50],
            div_style={"display": "flex", "flex-wrap": "wrap", "justify-content": "space-between", "height": "300px", "overflow-y": "auto", "padding": "5px", "gap": "5px"},
            img_style={"cursor": "pointer", "border-radius": "4px", "transition": "transform 0.2s", "border": "3px solid #333", "width": "30%", "height": "60px", "object-fit": "cover"},
            key="target_grid",
            default=tgt_idx_grid,
            border_color="#00ADB5"
        )
        
        if clicked_tgt > -1 and clicked_tgt != tgt_idx_grid:
            st.session_state.selected_target = all_files[clicked_tgt]
            st.rerun()

    st.markdown("<div style='margin-top:30px; font-size:0.8rem; color:#666; font-weight:bold; margin-bottom:5px;'>PARAMETER DECK</div>", unsafe_allow_html=True)
    
    # CSS to force tiny tertiary buttons
    st.markdown("""
    <style>
    div[data-testid="column"] button[kind="tertiary"] {
        padding: 0px 5px !important;
        min-height: 1.5rem !important;
        height: 1.5rem !important;
        line-height: 1 !important;
        border: none !important;
        margin-top: 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # helper for granular reset
    def ui_reset_item(label, min_v, max_v, default_v, step_v, key_s, help_s):
        # Header Row: Label + Small Reset Button
        c_label, c_btn = st.columns([0.9, 0.1]) 
        with c_label:
            st.markdown(f"**{label}**", help=help_s) 
        with c_btn:
            if st.button("↺", key=f"rst_{key_s}", help="Reset", type="tertiary"):
                st.session_state[key_s] = default_v
                st.rerun()
        # Slider
        return st.slider("", min_v, max_v, default_v, step_v, key=key_s, label_visibility="collapsed", help=help_s)

    def ui_reset_checkbox(label, default_v, key_s, help_s):
        # Header Row
        c_label, c_btn = st.columns([0.9, 0.1])
        with c_label:
             st.markdown(f"**{label}**")
        with c_btn:
            if st.button("↺", key=f"rst_{key_s}", help="Reset", type="tertiary"):
                st.session_state[key_s] = default_v
                st.rerun()
        # Checkbox
        return st.checkbox("", value=st.session_state.get(key_s, default_v), key=key_s, label_visibility="collapsed", help=help_s)
    
    # 2. PARAMETER DECK
    tab_norm, tab_grade, tab_fx = st.tabs(["NORM", "GRADE", "FX"])
    
    with tab_norm:
        st.caption("ALGORITHM PARAMETERS")
        io_val = ui_reset_item("Io (TRANSMISSION)", 150, 255, 255, 1, "p_io", help_s="**Intensità luce Trasmessa**")
        alpha = ui_reset_item("ALPHA (PERCENTILE)", 0.0, 5.0, 1.0, 0.1, "p_alpha", help_s="**Robustezza Stima Colore**")
        beta = ui_reset_item("BETA (OD THRESHOLD)", 0.0, 0.5, 0.15, 0.01, "p_beta", help_s="**Soglia di Trasparenza**")
        st.caption("MATCHING")
        use_lum_match = ui_reset_checkbox("Luminosity Match", True, "p_use_lum", help_s="**Standardizzazione Luminosità**\n\n- **A cosa serve**: Pareggia l'esposizione della foto sorgente a quella target PRIMA dell'analisi.\n- **Effetto**: Corregge foto scure/sovraesposte rendendole coerenti.")
        use_hist_match = ui_reset_checkbox("Histogram Match", True, "p_use_hist", help_s="**Matching Istogramma Colore**\n\n- **A cosa serve**: Forza la distribuzione dei colori (H&E) ad essere identica al target.\n- **Effetto**: Garantisce che i viola e i rosa siano esattamente della stessa tonalità del riferimento.")

    with tab_grade:
        st.caption("TONE CURVE (SPLINE)")
        # Dense Sliders
        p0 = ui_reset_item("BLACKS (0)", 0, 255, 0, 1, "p_p0", help_s="**Punto Neri**")
        p1 = ui_reset_item("SHADOWS (64)", 0, 255, 64, 1, "p_p1", help_s="**Punto Ombre**")
        p2 = ui_reset_item("MIDTONES (128)", 0, 255, 128, 1, "p_p2", help_s="**Punto Medi**")
        p3 = ui_reset_item("HIGHLIGHTS (192)", 0, 255, 192, 1, "p_p3", help_s="**Punto Luci**")
        p4 = ui_reset_item("WHITES (255)", 0, 255, 255, 1, "p_p4", help_s="**Punto Bianchi**")
        
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
        contrast = ui_reset_item("CONTRAST", 0.5, 2.0, 1.0, 0.05, "p_contrast", help_s="**Contrasto Lineare**")
        gamma = ui_reset_item("GAMMA", 0.1, 3.0, 1.0, 0.05, "p_gamma", help_s="**Correzione Gamma**")
        to_bw = ui_reset_checkbox("Monochrome Mode", False, "p_bw", help_s="Converte l'output finale in Bianco e Nero.")

    with tab_fx:
        st.caption("STEATOSIS ENHANCEMENT")
        clarify_strength = ui_reset_item("CLARIFIER STRENGTH", 0.0, 2.0, 1.0, 0.1, "p_clarify", help_s="**Filtro Pulizia Steatosi**")
        st.caption("DETAIL ENHANCEMENT")
        sharpness = ui_reset_item("UNSHARP MASK", 0.0, 3.0, 1.0, 0.1, "p_sharpness", help_s="**Nitidezza (Unsharp Mask)**")

    st.markdown("---")
    if st.button("PROCESS BATCH", type="primary", use_container_width=True):
         st.session_state.run_batch_trigger = True


# --- MAIN VIEWPORT ---
# Top Bar
if 'selected_sample' not in st.session_state:
     st.session_state.selected_sample = all_files[1] if len(all_files)>1 else all_files[0]

# --- PROCESSED GALLERY ---
try:
    processed_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
except FileNotFoundError:
    processed_files = []

if processed_files:
    with st.expander("PROCESSED GALLERY (VERTICAL VIEW)", expanded=False):
        st.caption("📂 **FULL-SIZE REVIEW**", help="Scorri verticalmente per analizzare i risultati in dettaglio.")
        
        # Load processed thumbnails (LARGER for detail review)
        with st.spinner("Loading high-res previews..."):
            proc_thumbnails = get_thumbnails(processed_files[:20], OUTPUT_DIR, size=(600, 600))
            
        clicked_proc = clickable_images(
            proc_thumbnails,
            titles=processed_files[:20],
            div_style={
                "display": "flex", 
                "flex-direction": "column", 
                "height": "600px", 
                "overflow-y": "auto", 
                "gap": "30px", 
                "padding": "15px",
                "align-items": "center"
            },
            img_style={
                "cursor": "pointer", 
                "border-radius": "8px", 
                "border": "3px solid #333", 
                "width": "100%", 
                "max-width": "600px", 
                "height": "auto", 
                "object-fit": "contain"
            },
            key="processed_gallery"
        )
        if clicked_proc > -1:
             st.info(f"Viewing: {processed_files[clicked_proc]}")

# --- IMAGE TRAY CAROUSEL ---
with st.expander("IMAGE TRAY", expanded=True):
    st.caption("🎞️ **FILMSTRIP CAROUSEL**", help="**Image Carousel**\nClicca su un'immagine per caricarla. Scorri orizzontalmente con Shift+Scroll o Touch.")
    
    # 1. Use Global Thumbnails (already computed)
    tray_files = all_files[:50] # Matches global prep
    
    # 2. Render Clickable Carousel
    # Highlight active
    curr_idx = tray_files.index(st.session_state.selected_sample) if st.session_state.selected_sample in tray_files else -1
    
    clicked = clickable_images(
        all_thumbnails,  # Reuse global cache 
        titles=tray_files,
        div_style={"display": "flex", "justify-content": "flex-start", "flex-wrap": "nowrap", "overflow-x": "auto", "padding": "10px", "gap": "10px"},
        img_style={"cursor": "pointer", "border-radius": "4px", "transition": "transform 0.2s", "border": "5px solid #444", "height": "100px", "min-width": "100px", "object-fit": "cover"},
        key="tray_carousel",
        default=curr_idx,
        border_color="#00ADB5"
    )
    
    # 4. Handle Selection
    if clicked > -1 and clicked != curr_idx:
         st.session_state.selected_sample = tray_files[clicked]
         st.rerun()

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



# Batch Runner
if st.session_state.get('run_batch_trigger', False):
    st.info("BATCH JOB STARTED...")
    pbar = st.progress(0)
    
    # Prepare Metadata (Static for this batch)
    metadata = PngInfo()
    metadata.add_text("Macenko_Target", st.session_state.selected_target)
    metadata.add_text("Macenko_Params", json.dumps({"alpha": alpha, "beta": beta, "io": io_val}))
    metadata.add_text("PostProcess", json.dumps({"gamma": gamma, "contrast": contrast, "sharpness": sharpness, "clarifier": clarify_strength}))
    metadata.add_text("Curve", str(curve_points))
    metadata.add_text("LuminosityMatch", str(use_lum_match))
    metadata.add_text("HistogramMatch", str(use_hist_match))

    for i, f in enumerate(all_files):
        try:
             src = np.array(Image.open(os.path.join(IMAGE_DIR, f)).convert('RGB'))
             out = normalization_core.process_image(
                t_img, src, # target is static from current
                use_lum_match, use_hist_match,
                clarify_strength, sharpness, contrast, gamma,
                curve_points, to_bw, alpha, beta, io_val
             )
             # Force PNG for metadata support
             out_name = os.path.splitext(f)[0] + ".png"
             out.save(os.path.join(OUTPUT_DIR, out_name), pnginfo=metadata)
        except Exception as e:
             print(f"Error processing {f}: {e}")
             pass
        pbar.progress((i+1)/len(all_files))
        
    st.success("BATCH COMPLETE")
    st.session_state.run_batch_trigger = False

