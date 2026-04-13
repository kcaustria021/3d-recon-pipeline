import numpy as np

def npy_to_ply(npy_path, ply_path):
    points = np.load(npy_path)  # shape (N, 3)
    N = len(points)

    with open(ply_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Points
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

npy_to_ply("media/testing/test_torus/torus_recon_halfspace.npy", "media/testing/test_torus/torus_recon_halfspace.ply")