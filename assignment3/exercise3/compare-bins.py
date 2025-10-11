import sys
import numpy as np

if len(sys.argv) != 3:
    print("usage: python compare_bins.py file1.bin file2.bin")
    sys.exit(1)

a = np.fromfile(sys.argv[1], dtype=np.float64)
b = np.fromfile(sys.argv[2], dtype=np.float64)

if a.size != b.size:
    print("Different size:", a.size, b.size)
    sys.exit(2)

diff = np.abs(a - b)
max_abs = diff.max()
mean_abs = diff.mean()
rel = diff / np.maximum(np.abs(a), 1e-300)  # fixes division by zero bug
max_rel = rel.max()

print("elements:", a.size)
print("max abs diff:", max_abs)
print("mean abs diff:", mean_abs)
print("max rel diff:", max_rel)
