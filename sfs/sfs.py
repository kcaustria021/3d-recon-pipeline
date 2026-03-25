import numpy as np
import cv2
import glob

from utils_sfs import compute_proj_matrix
from utils_sfs import View
from utils_sfs import Node
from utils_sfs import carve_voxels
from utils_sfs import compute_bounds
from utils_sfs import cubify
from utils_sfs import export_to_ply

def main():
    imgs = glob.glob("media/testing/objs/*.png")
    masks = glob.glob("media/testing/masks/*.png")

    assert len(imgs) == len(masks), "Number of images must match number of masks"
    n_views = len(imgs)

    # initialize views
    views = []
    for i in range(n_views):
        print(f"Initialize View {i}")
        dist, P = compute_proj_matrix(i)
        view = View(imgs[i], masks[i], P, dist)
        views.append(view)

    min_bound, max_bound = compute_bounds(views)
    cube_min, cube_max = cubify(min_bound, max_bound)
    x_min, y_min, z_min = cube_min
    x_max, y_max, z_max = cube_max

    # initialize octree
    octree_root = Node(bounds=(x_min, x_max, y_min, y_max, z_min, z_max), max_depth=6)
    print("Carving voxels...")
    carve_voxels(octree_root, views)

    export_to_ply(octree_root, "media/testing/recon.ply")

if __name__ == "__main__":
    main()
