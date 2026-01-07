
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from normalization import norm_macenko, get_stain_vectors, get_concentrations

def get_mean_lab_vector(img_path):
    """
    Reads an image, converts to Lab, returns the global mean [L, a, b].
    """
    try:
        # Load in RGB
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Convert to Lab
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Compute mean
        mean_vector = np.mean(lab, axis=(0, 1))
        return mean_vector
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def compute_centroid_and_distances(features):
    """
    features: dict {filename: np.array([L, a, b])}
    Returns:
        centroid: np.array([L, a, b])
        distances: list of (filename, distance) sorted by distance
    """
    vectors = np.array(list(features.values()))
    
    # Dataset Centroid
    centroid = np.mean(vectors, axis=0)
    
    # Distances
    distances = []
    for fname, vec in features.items():
        dist = np.linalg.norm(vec - centroid)
        distances.append((fname, dist))
        
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    return centroid, distances

def get_folder_coherence_score(folder_path):
    """
    Phase 3: Quality Metric (Post-Normalization Scoring)
    1. Extract mean CIELAB vectors for all images.
    2. Compute Covariance Matrix.
    3. Score = Trace(Covariance).
    """
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not files:
        return float('inf')
        
    vectors = []
    for f in files:
        path = os.path.join(folder_path, f)
        vec = get_mean_lab_vector(path)
        if vec is not None:
            vectors.append(vec)
            
    if len(vectors) < 2:
        return float('inf')
        
    vectors_np = np.array(vectors)
    
    # Covariance Matrix
    cov_matrix = np.cov(vectors_np, rowvar=False)
    
    # Score = Trace (Total Variance)
    score = np.trace(cov_matrix)
    return score

def main():
    parser = argparse.ArgumentParser(description="Stain Normalization Optimization Pipeline")
    parser.add_argument('--image_dir', type=str, default='./image', help="Input directory containing images")
    parser.add_argument('--output_dir', type=str, default='./optimization_results', help="Root directory for output")
    parser.add_argument('--beta', type=float, default=0.15, help="OD threshold")
    parser.add_argument('--alpha', type=float, default=1.0, help="Percentile")
    
    args = parser.parse_args()
    
    # Ensure input exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Input directory {args.image_dir} does not exist.")
        return

    # Ensure output exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    supported_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    all_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(supported_exts)]
    all_files.sort()
    
    print(f"Found {len(all_files)} images in {args.image_dir}")
    
    # --- Phase 1: Candidate Selection ---
    print("\n--- Phase 1: Candidate Selection (Pre-Normalization Scoring) ---")
    features = {}
    
    print("Computing feature vectors (Mean Lab)...")
    for f in tqdm(all_files):
        path = os.path.join(args.image_dir, f)
        vec = get_mean_lab_vector(path)
        if vec is not None:
            features[f] = vec
            
    centroid, distances = compute_centroid_and_distances(features)
    
    # Select Top 14 Candidates (Medoids)
    top_candidates = distances[:14]
    print("\nTop 14 Candidates (Medoids):")
    for i, (fname, dist) in enumerate(top_candidates):
        print(f"  {i+1}. {fname} (Dist: {dist:.4f})")
        
    # --- Phase 2: Mass Normalization Sweep ---
    print("\n--- Phase 2: Mass Normalization Sweep ---")
    
    normalization_runs = []
    
    for i, (target_fname, dist) in enumerate(top_candidates):
        run_id = i + 1
        run_dir_name = f"norm_run_{run_id:02d}"
        run_output_dir = os.path.join(args.output_dir, run_dir_name)
        
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)
            
        print(f"\nRunning Normalization Sweep {run_id}/14 using target: {target_fname}")
        
        # 1. Extract Stats for this target ONCE
        target_path = os.path.join(args.image_dir, target_fname)
        try:
            target_img = np.array(Image.open(target_path).convert('RGB'))
            
            # Lump computation
            t_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
            t_l = t_lab[:,:,0]
            t_tissue = t_l[t_l < 220]
            target_median_lum = np.median(t_tissue) if len(t_tissue) > 0 else 150
            
            target_stain_mat = get_stain_vectors(target_img, args.beta, args.alpha)
            target_concs = get_concentrations(target_img, target_stain_mat, args.beta)
            target_max_c = np.percentile(target_concs, 99, axis=0)
            
        except Exception as e:
            print(f"FATAL: Could not extract stats from candidate {target_fname}: {e}")
            continue
            
        # 2. Normalize entire dataset
        for f in tqdm(all_files, desc=f"Run {run_id} ({target_fname})"):
            src_path = os.path.join(args.image_dir, f)
            dst_path = os.path.join(run_output_dir, f)
            
            try:
                norm_macenko(
                    source_img_path=src_path,
                    save_path=dst_path,
                    beta=args.beta,
                    alpha=args.alpha,
                    target_img_path=None, # Not used since we pass stats
                    target_stain_matrix=target_stain_mat,
                    target_max_C=target_max_c,
                    target_tissue_median_lum=target_median_lum
                )
            except Exception as e:
                # print(f"Failed token {f}: {e}")
                pass
                
        normalization_runs.append({
            'run_id': run_id,
            'folder': run_output_dir,
            'target': target_fname
        })

    # --- Phase 3 & 4: Scoring and Winner Selection ---
    print("\n--- Phase 3 & 4: Quality Metric & Winner Selection ---")
    
    results = []
    
    for run in normalization_runs:
        print(f"Scoring Run {run['run_id']} ({run['target']})...")
        score = get_folder_coherence_score(run['folder'])
        run['score'] = score
        results.append(run)
        print(f"  Score (Total Variance): {score:.4f}")
        
    # Sort by Score (Ascending)
    results.sort(key=lambda x: x['score'])
    
    print("\n\n=== OPTIMIZATION RESULTS (Ranking) ===")
    print(f"{'Rank':<5} | {'Score (Data Var)':<18} | {'Target Image':<30} | {'Folder'}")
    print("-" * 80)
    
    for rank, res in enumerate(results):
        print(f"{rank+1:<5} | {res['score']:<18.4f} | {res['target']:<30} | {os.path.basename(res['folder'])}")
        
    winner = results[0]
    print(f"\n\nWINNER: {winner['target']} (Score: {winner['score']:.4f})")
    print(f"Best Normalized Dataset located in: {winner['folder']}")
    
    # Save results to text file
    with open(os.path.join(args.output_dir, 'ranking_results.txt'), 'w') as f:
        f.write("=== OPTIMIZATION RESULTS ===\n")
        f.write(f"Winner: {winner['target']} (Score: {winner['score']:.4f})\n\n")
        f.write(f"{'Rank':<5} | {'Score':<18} | {'Target Image'}\n")
        for rank, res in enumerate(results):
            f.write(f"{rank+1:<5} | {res['score']:<18.4f} | {res['target']}\n")

if __name__ == "__main__":
    main()
