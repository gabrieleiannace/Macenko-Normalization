import numpy as np
from PIL import Image, ImageEnhance
import cv2
from skimage import exposure
import os
from scipy.interpolate import PchipInterpolator

def get_stain_vectors(I, beta=0.15, alpha=1):
    I = I.reshape((-1, 3))
    OD = -np.log10((I.astype(np.float64) + 1) / 255.0)
    ODhat = OD[np.all(OD > beta, axis=1)]
    if len(ODhat) < 10:
        return np.eye(2), np.zeros(2)
    _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))
    Vec = V[:, 1:3]
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])
    return HE.T

def get_concentrations(I, stain_matrix, beta=0.15):
    I = I.reshape((-1, 3))
    OD = -np.log10((I.astype(np.float64) + 1) / 255.0)
    C = np.dot(OD, np.linalg.pinv(stain_matrix.T))
    return C

def match_luminosity(source_rgb, target_rgb):
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)
    src_l, src_a, src_b = cv2.split(src_lab)
    tgt_l, tgt_a, tgt_b = cv2.split(tgt_lab)
    src_mean, src_std = np.mean(src_l), np.std(src_l)
    tgt_mean, tgt_std = np.mean(tgt_l), np.std(tgt_l)
    src_l = src_l.astype(np.float32)
    l_matched = ((src_l - src_mean) * (tgt_std / (src_std + 1e-5))) + tgt_mean
    l_matched = np.clip(l_matched, 0, 255).astype(np.uint8)
    matched_lab = cv2.merge((l_matched, src_a, src_b))
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def clarify_steatosis(img_rgb, strength=1.0):
    # Strength factor for UI tuning
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    highlight_mask = np.clip((v - 200) / 55.0, 0, 1)
    
    # Apply strength
    desat_factor = 0.9 * strength
    val_boost_factor = 0.8 * strength
    
    s = s * (1.0 - highlight_mask * desat_factor)
    v = v + (255 - v) * highlight_mask * val_boost_factor
    
    hsv_new = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_custom_curve(img_rgb, points):
    """
    Applies a custom tone curve defined by points [(x0, y0), (x1, y1), ...].
    Uses PCHIP interpolation for smooth monotonic curves.
    """
    # Sort points by X just in case
    points.sort(key=lambda p: p[0])
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    
    # Interpolate to create LUT (0-255)
    interpolator = PchipInterpolator(x_points, y_points)
    x_lut = np.arange(256)
    y_lut = interpolator(x_lut)
    
    # Clip to valid range
    y_lut = np.clip(y_lut, 0, 255).astype(np.uint8)
    
    # Apply LUT
    # We apply this to all channels (RGB) equally to affect Tone/Luminosity without major color shifts,
    # or per-channel if requested. For "Curve" usually it's Master Channel (all).
    return cv2.LUT(img_rgb, y_lut)

def process_image(target_img_np, source_img_np, 
                 use_lum_match=True, 
                 use_hist_match=True, 
                 clarify_strength=1.0, 
                 sharpness=1.0,
                 contrast=1.0,
                 gamma=1.0,
                 curve_points=None,
                 to_bw=False,
                 alpha=1.0, beta=0.15, io=255):
    
    Io = io
    
    # 1. Luminosity Match
    if use_lum_match:
        source_img_np = match_luminosity(source_img_np, target_img_np)
        
    h, w, c = source_img_np.shape
    
    # 2. Macenko
    target_stain_matrix = get_stain_vectors(target_img_np, beta, alpha)
    target_concentrations = get_concentrations(target_img_np, target_stain_matrix, beta)
    
    source_stain_matrix = get_stain_vectors(source_img_np, beta, alpha)
    source_concentrations = get_concentrations(source_img_np, source_stain_matrix, beta)
    
    if use_hist_match:
        C_norm = np.zeros_like(source_concentrations)
        C_norm[:, 0] = exposure.match_histograms(source_concentrations[:, 0], target_concentrations[:, 0])
        C_norm[:, 1] = exposure.match_histograms(source_concentrations[:, 1], target_concentrations[:, 1])
    else:
        # Standard Percentile
        target_max_C = np.percentile(target_concentrations, 99, axis=0)
        source_max_C = np.percentile(source_concentrations, 99, axis=0)
        source_max_C[source_max_C == 0] = 1e-5
        C_norm = source_concentrations * (target_max_C / source_max_C)
        
    OD_norm = np.dot(C_norm, target_stain_matrix.T)
    I_norm = Io * np.exp(-OD_norm)
    I_norm = np.clip(I_norm, 0, 255).astype(np.uint8)
    I_norm = I_norm.reshape((h, w, c))
    
    # 3. Clarification
    if clarify_strength > 0:
        I_norm = clarify_steatosis(I_norm, strength=clarify_strength)
        
    # 4. Sharpening
    if sharpness != 1.0:
        amount = 0.8 * sharpness
        gaussian = cv2.GaussianBlur(I_norm, (0, 0), 1.0)
        I_norm = cv2.addWeighted(I_norm, 1.0 + amount, gaussian, -amount, 0)
    
    # 5. Contrast/Gamma/Curve
    # Apply Custom Curve FIRST (most powerful)
    if curve_points is not None and len(curve_points) >= 2:
        I_norm = apply_custom_curve(I_norm, curve_points)

    # Gamma (Standard)
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        I_norm = cv2.LUT(I_norm, table)

    # Contrast (Linear)
    if contrast != 1.0:
        I_norm = I_norm.astype(np.float32)
        I_norm = (I_norm - 127.0) * contrast + 127.0
        I_norm = np.clip(I_norm, 0, 255).astype(np.uint8)
        
    # 6. B/W Mode
    if to_bw:
        I_norm = cv2.cvtColor(I_norm, cv2.COLOR_RGB2GRAY)
        I_norm = cv2.cvtColor(I_norm, cv2.COLOR_GRAY2RGB) 
        
    return Image.fromarray(I_norm)
