import os
import subprocess
import numpy as np
import find_best_target # Reusing analysis logic
from tqdm import tqdm
import random

# Top 14 Candidates from previous analysis
CANDIDATES = [
    "1004553_27.png",
    "1001343_53.png",
    "1004492_6.png",
    "1001542_89.png",
    "1002644_96.png",
    "1003563_31.png",
    "1001939_83.png",
    "1001015_87.png",
    "1001564_73.png",
    "1004614_92.png",
    "1004545_51.png",
    "1004122_72.png",
    "1002564_4.png",
    "1001998_48.png"
]

INPUT_DIR = "image"
BASE_OUTPUT_DIR = "experiments"
PYTHON_EXE = "./.venv/bin/python"

def run_experiment():
    # SET SEED FOR DETERMINISM
    random.seed(42)
    np.random.seed(42)

    folder_scores = []
    
    print(f"Starting comparison of {len(CANDIDATES)} targets...")
    print("Scoring Formula: Score = (1.5 * Sep) + (2.0 * Nuclei_Contrast) + (1.0 * Eosin_Int) + (2.0 * Background_Lum)")
    
    for rank, target_file in enumerate(CANDIDATES):
        target_name = os.path.splitext(target_file)[0]
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"target_{target_name}")
        target_path = os.path.join(INPUT_DIR, target_file)
        
        print(f"\n[{rank+1}/{len(CANDIDATES)}] Processing Target: {target_file}")
        
        # 1. Check if we need to run normalization
        # Strict check: exist AND have files
        should_run = True
        if os.path.exists(output_dir):
            existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            if len(existing_files) > 400:
                print(f"   > Skipping normalization (already exists with {len(existing_files)} files)")
                should_run = False
        
        if should_run:
            cmd = [
                PYTHON_EXE, "normalization.py",
                "--target", target_path,
                "--input_dir", INPUT_DIR,
                "--output_dir", output_dir
            ]
            try:
                # Run with output ignored to prevent buffer deadlock
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Error iterating {target_file}: {e}")
                continue
            
        # 2. Analyze Results
        print(f"   > Analyzing output quality in {output_dir}...")
        
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
        
        # Deterministic Sampling
        # Sort first to ensure 'random.sample' picks the same file set every time given the same seed
        files.sort()
        if len(files) > 50:
            files = random.sample(files, 50)
        
        batch_results = []
        for img_path in tqdm(files, leave=False):
            res = find_best_target.analyze_image(img_path)
            if res:
                batch_results.append(res)
        
        if not batch_results:
            print("   > No valid results found.")
            continue
            
        # Score calculation explanation:
        # We process the batch through the same scoring logic
        scored_batch = find_best_target.score_metrics(batch_results)
        
        # Calculate Folder Averages
        avg_score = np.mean([r['score'] for r in scored_batch])
        avg_sep = np.mean([r['stain_sep_deg'] for r in scored_batch])
        avg_h = np.mean([r['h_99'] for r in scored_batch])
        avg_e = np.mean([r['e_99'] for r in scored_batch])
        avg_bg = np.mean([r['bg_lum_mean'] for r in scored_batch])
        
        folder_summary = {
            "target": target_file,
            "folder": output_dir,
            "avg_score": avg_score,
            "avg_sep": avg_sep,
            "avg_h": avg_h,
            "avg_e": avg_e,
            "avg_bg": avg_bg
        }
        folder_scores.append(folder_summary)
        
        print(f"   > Folder Score: {avg_score:.2f} (Sep: {avg_sep:.1f}, BG: {avg_bg:.1f})")

    # 3. Final Ranking
    print("\n\n" + "="*80)
    print("FINAL EXPERIMENT REPORT: TARGET COMPARISON")
    print("="*80)
    print(f"{'Rank':<5} {'Target Image':<25} {'AvgScore':<10} {'AvgSep':<8} {'AvgH':<8} {'AvgE':<8} {'AvgBG':<8}")
    print("-" * 80)
    
    sorted_folders = sorted(folder_scores, key=lambda x: x['avg_score'], reverse=True)
    
    for i, f in enumerate(sorted_folders):
        print(f"{i+1:<5} {f['target']:<25} {f['avg_score']:.3f}      {f['avg_sep']:.1f}     {f['avg_h']:.2f}     {f['avg_e']:.2f}     {f['avg_bg']:.1f}")
        
    # Save report
    with open("experiment_ranking.txt", "w") as f:
        f.write("Rank,Target,AvgScore,AvgSep,AvgH,AvgE,AvgBG\n")
        for i, row in enumerate(sorted_folders):
            f.write(f"{i+1},{row['target']},{row['avg_score']},{row['avg_sep']},{row['avg_h']},{row['avg_e']},{row['avg_bg']}\n")

if __name__ == "__main__":
    run_experiment()
