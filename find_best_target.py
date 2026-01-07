import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import normalization  # Importing the user's existing normalization logic

def calculate_angle(v1, v2):
    """Calculate angle in degrees between two vectors."""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0: return 0
    cos_angle = dot_product / norms
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_image(img_path):
    try:
        # Load and resize for speed
        img = Image.open(img_path).convert('RGB')
        
        # Resize to max dimension 1024 to speed up SVD and pixel processing
        # Macenko vectors are stain-dependent, not scale-dependent (mostly)
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        
        img_np = np.array(img)
        
        # 1. Get Stain Vectors
        # Returns HE.T shape (2, 3) -> Rows are stain vectors?
        # normalization.py: return HE.T where HE was np.array([v1, v2]). 
        # So shape is (3, 2) ? 
        # Let's check normalization.py again.
        # Line 62: return HE.T. HE is constructed as np.array([vMin, vMax]). 
        # vMin is shape (3,). So HE is (2, 3). HE.T is (3, 2).
        # Columns are the stain vectors.
        target_stain_matrix = normalization.get_stain_vectors(img_np)
        
        # Stain Separation (Angle between H and E)
        # H = target_stain_matrix[:, 0]
        # E = target_stain_matrix[:, 1]
        v1 = target_stain_matrix[:, 0]
        v2 = target_stain_matrix[:, 1]
        separation_angle = calculate_angle(v1, v2)
        
        # 2. Get Concentrations
        C = normalization.get_concentrations(img_np, target_stain_matrix)
        # C is (N, 2)
        
        # Nuclei (H) Robust Max Intensity (99th percentile)
        # High means distinct nuclei
        h_99 = np.percentile(C[:, 0], 99)
        
        # Cytoplasm (E) Robust Max Intensity
        # "Alive but not saturated"
        e_99 = np.percentile(C[:, 1], 99)
        
        # 3. Background Analysis
        # "Bucchi bianchi devono essere ben contrastati"
        # Convert to LAB for Luminosity analysis
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Identify background: High Luminosity (> 220 is a safe bet for raw slides)
        # Or simple OD threshold.
        # normalization.py uses OD > beta to find tissue.
        # Let's use OD < beta for background.
        img_reshaped = img_np.reshape((-1, 3))
        OD = -np.log10((img_reshaped.astype(float) + 1) / 255.0)
        od_max = np.max(OD, axis=1)
        background_mask = od_max < 0.15 # beta default is 0.15
        
        if np.sum(background_mask) > 0:
            bg_pixels_l = l_channel.reshape(-1)[background_mask]
            bg_luminosity_mean = np.mean(bg_pixels_l)
            bg_luminosity_std = np.std(bg_pixels_l) # Low std means uniform clean white
        else:
            bg_luminosity_mean = 0
            bg_luminosity_std = 100
            
        return {
            "filename": os.path.basename(img_path),
            "stain_sep_deg": separation_angle,
            "h_99": h_99,
            "e_99": e_99,
            "bg_lum_mean": bg_luminosity_mean,
            "bg_lum_std": bg_luminosity_std
        }
    except Exception as e:
        return None

def score_metrics(results):
    # Convert to standard arrays
    seps = np.array([r['stain_sep_deg'] for r in results])
    h_99s = np.array([r['h_99'] for r in results])
    e_99s = np.array([r['e_99'] for r in results])
    bg_lums = np.array([r['bg_lum_mean'] for r in results])
    
    # Normalize (0-1)
    norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)
    
    n_sep = norm(seps)
    n_h = norm(h_99s)
    n_e = norm(e_99s)
    n_bg = norm(bg_lums)
    
    # Heuristic Scores
    
    # 1. Separation: Higher is better (distinct colors)
    score_sep = n_sep 
    
    # 2. Nuclei: Higher is better (distinct)
    score_h = n_h
    
    # 3. Eosin: "Alive but not saturated"
    # We punish the very bottom (dead) and maybe slightly punish the absolute max if it looks like outlier?
    # Actually, usually higher is better for "Alive", unless it's an artifact.
    # Let's assume higher E is good, but we verify visually.
    score_e = n_e 
    
    # 4. Background: Higher Mean L is better (whiter holes).
    score_bg = n_bg
    
    # Weights
    # Distinct Nuclei is critical (Viola distinto)
    # Background contrast is critical (Bucchi bianchi ben contrastati)
    
    final_scores = (
        1.5 * score_sep + 
        2.0 * score_h + 
        1.0 * score_e + 
        2.0 * score_bg
    )
    
    for i, r in enumerate(results):
        r['score'] = final_scores[i]
        
    return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='image')
    args = parser.parse_args()
    
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    print(f"Scanning {len(files)} images...")
    
    results = []
    for f in tqdm(files):
        res = analyze_image(f)
        if res: results.append(res)
        
    ranked = score_metrics(results)
    
    print("\nTop 5 Candidates for Target Image:")
    print(f"{'Rank':<5} {'Filename':<30} {'Score':<8} {'Sep(deg)':<10} {'H_99':<8} {'E_99':<8} {'BG_Lum':<8}")
    print("-" * 90)
    for i, r in enumerate(ranked[:15]):
        print(f"{i+1:<5} {r['filename']:<30} {r['score']:.2f}     {r['stain_sep_deg']:.1f}       {r['h_99']:.2f}     {r['e_99']:.2f}     {r['bg_lum_mean']:.1f}")
        
    # Save top 1 path to a file for easy retrieval
    with open("suggested_target.txt", "w") as f:
        f.write(ranked[0]['filename'])
