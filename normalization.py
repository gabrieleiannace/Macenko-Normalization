import numpy as np
from PIL import Image
import cv2
from skimage import exposure
import os
import argparse

def get_stain_vectors(I, beta=0.15, alpha=1):
    """
    Get the stain matrix (2x3) using Macenko's method.
    """
    # Reshape to (H*W, 3)
    I = I.reshape((-1, 3))

    # Calculate Optical Density (OD)
    # OD = -log10((I+1)/Io) -> we assume Io=255 for standard RGB 8-bit images
    # Add 1 to avoid log(0)
    OD = -np.log10((I.astype(np.float64) + 1) / 255.0)

    # Remove data with too little absorbance (transparent pixels)
    ODhat = OD[np.all(OD > beta, axis=1)]
    
    if len(ODhat) < 10:
        # Fallback if almost empty
        return np.eye(2), np.zeros(2)

    # PCA via SVD
    # Covariance matrix
    # Normalize features
    _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))

    # Project onto the regular plane
    # V[:, 1:] corresponds to the eigenvectors corresponding to the two largest eigenvalues
    # Note: numpy eigh returns eigenvalues in ascending order, so take last 2.
    Vec = V[:, 1:3] # Shape (3, 2). These are the directions.

    # Project data onto the plane
    That = np.dot(ODhat, Vec)

    # Find the min and max vectors (robustly)
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Heuristic: make sure the first vector is H (more pink/red) and second is E (more blue/purple)
    # H usually has higher red component relative to blue in OD space?
    # OD varies. Hematoxylin stains blue/purple, Eosin stains pink.
    # In RGB: H is dark blue/purple (low Red/Green, mid Blue). E is pink (high Red, low Green/Blue).
    # In OD: H absorbs Red helps. E absorbs Green.
    # Typically we allow arbitrary order, but for consistency we sort.
    # Let's just return them sorted by the angle for now to ensure consistency.

    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])
    
    # Rows are stain vectors. Shape (2, 3)
    return HE.T

def get_concentrations(I, stain_matrix, beta=0.15):
    """
    Get concentrations given an image and stain matrix.
    """
    I = I.reshape((-1, 3))
    OD = -np.log10((I.astype(np.float64) + 1) / 255.0)
    
    # We solve OD = C * S.  C is (N, 2), S is (2, 3).
    # C = OD * pinv(S)
    
    C = np.dot(OD, np.linalg.pinv(stain_matrix.T)) # stain_matrix is (3, 2) here technically based on above return
    # Wait, my previous return was HE.T which is (3, 2).
    # so we want stain_matrix to be (3, 2).
    
    return C

def match_luminosity(source_rgb, target_rgb):
    """
    Matches the luminosity (L channel in LAB) of the source image to the target image.
    This ensures consistent brightness/contrast before stain normalization.
    """
    # Convert to LAB
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)
    
    # Split channels
    src_l, src_a, src_b = cv2.split(src_lab)
    tgt_l, tgt_a, tgt_b = cv2.split(tgt_lab)
    
    # Calculate stats
    src_mean, src_std = np.mean(src_l), np.std(src_l)
    tgt_mean, tgt_std = np.mean(tgt_l), np.std(tgt_l)
    
    # Normalize Source L to match Target L stats
    # L_new = (L_src - mean_src) * (std_tgt / std_src) + mean_tgt
    # Epsilon to avoid div by zero
    src_l = src_l.astype(np.float32)
    l_matched = ((src_l - src_mean) * (tgt_std / (src_std + 1e-5))) + tgt_mean
    
    # Clip to 0-255
    l_matched = np.clip(l_matched, 0, 255).astype(np.uint8)
    
    # Merge back
    matched_lab = cv2.merge((l_matched, src_a, src_b))
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    
    return matched_rgb

def clarify_steatosis(img_rgb):
    """
    Specifically targets bright areas (potential steatosis) to make them:
    1. Pure White (Increase Value)
    2. Colorless (Decrease Saturation)
    
    This creates the "clean holes" look without affecting tissue color.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Define a soft mask for "Highlights"
    # We want pixels that are already quite bright (e.g. V > 200)
    # 200 -> 0.0, 255 -> 1.0
    highlight_mask = np.clip((v - 200) / 55.0, 0, 1)
    
    # 1. Desaturate Highlights
    # If mask is 1 (bright), saturation becomes 0.
    # If mask is 0 (dark/mid), saturation is unchanged.
    s = s * (1.0 - highlight_mask * 0.9) # Reduce saturation by 90% in highlights
    
    # 2. Boost Value in Highlights
    # Push bright pixels towards 255
    v = v + (255 - v) * highlight_mask * 0.8 # Close gap to white by 80%
    
    # Merge back
    hsv_new = cv2.merge((h, s, v))
    img_new = cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return img_new

def norm_macenko(target_img_path, source_img_path, save_path, Io=255, beta=0.15, alpha=1):
    # 1. Load target image
    target = np.array(Image.open(target_img_path).convert('RGB'))
    
    # 2. Get target stain matrix and max concentrations
    target_stain_matrix = get_stain_vectors(target, beta, alpha)
    target_concentrations = get_concentrations(target, target_stain_matrix, beta)
    target_max_C = np.percentile(target_concentrations, 99, axis=0)
    
    # 3. Load source image
    source = np.array(Image.open(source_img_path).convert('RGB'))
    h, w, c = source.shape
    
    # 4. Get source stain matrix
    source_stain_matrix = get_stain_vectors(source, beta, alpha)
    
    # 5. Extract source concentrations
    source_concentrations = get_concentrations(source, source_stain_matrix, beta)
    
    # 6. Normalize concentrations via Histogram Matching (User Request: "Stessi colori sempre")
    # Instead of linear scaling (Percentile), we match the entire distribution of staining intensities.
    # This ensures that the source image adopts the exact tonal balance (Dark/Mid/Light ratios) of the target.
    
    # Hematoxylin (Column 0) matching
    # match_histograms returns the matched array.
    C_norm = np.zeros_like(source_concentrations)
    C_norm[:, 0] = exposure.match_histograms(source_concentrations[:, 0], target_concentrations[:, 0])
    
    # Eosin (Column 1) matching
    C_norm[:, 1] = exposure.match_histograms(source_concentrations[:, 1], target_concentrations[:, 1])
    
    # 7. Reconstruct image using TARGET stain matrix
    OD_norm = np.dot(C_norm, target_stain_matrix.T)
    I_norm = Io * np.exp(-OD_norm)
    I_norm = np.clip(I_norm, 0, 255).astype(np.uint8)
    I_norm = I_norm.reshape((h, w, c))
    
    # --- PHASE 2: Final Polish (Clarification) ---
    # Apply the specific filter to whiten steatosis
    img_clarified = clarify_steatosis(I_norm)
    
    # Mild Sharpening (Radius 1.0) for definition
    gaussian = cv2.GaussianBlur(img_clarified, (0, 0), 1.0)
    unsharp = cv2.addWeighted(img_clarified, 1.8, gaussian, -0.8, 0)
    
    final_output = np.clip(unsharp, 0, 255).astype(np.uint8)
    
    img_out = Image.fromarray(final_output)
    img_out.save(save_path)
    print(f"Saved normalized image to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Macenko Stain Normalization")
    parser.add_argument('--target', type=str, required=True, help="Path to the reference image")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory of images to normalize")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save normalized images")
    parser.add_argument('--alpha', type=float, default=1.0, help="Percentile for stain vector estimation")
    parser.add_argument('--beta', type=float, default=0.15, help="OD threshold for transparency")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    supported_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    # Process all files
    files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in supported_exts]
    print(f"Found {len(files)} images.")
    
    for f in files:
        src = os.path.join(args.input_dir, f)
        dst = os.path.join(args.output_dir, f)
        try:
            norm_macenko(args.target, src, dst, alpha=args.alpha, beta=args.beta)
        except Exception as e:
            print(f"Failed to normalize {f}: {e}")
