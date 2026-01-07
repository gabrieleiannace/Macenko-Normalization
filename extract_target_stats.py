import numpy as np
from PIL import Image
import normalization

# Load target
target_path = "image/1004553_27.png"
img = np.array(Image.open(target_path).convert('RGB'))

# Get vectors and concentrations
stain_matrix = normalization.get_stain_vectors(img)
C = normalization.get_concentrations(img, stain_matrix)
max_C = np.percentile(C, 99, axis=0)

# Get Luminosity Median for brightness normalization
import cv2
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l = lab[:,:,0]
tissue_l = l[l < 220]
median_l = np.median(tissue_l)

print("TARGET_STAIN_MATRIX = np.array([")
for row in stain_matrix:
    print(f"    [{row[0]:.5f}, {row[1]:.5f}],")
print("])")

print(f"TARGET_MAX_C = np.array([{max_C[0]:.5f}, {max_C[1]:.5f}])")
print(f"TARGET_MEDIAN_LUM = {median_l:.1f}")
