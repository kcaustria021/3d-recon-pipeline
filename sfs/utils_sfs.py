import numpy as np
from numpy import typing as npt
import cv2
import os

class View:
    """
    Class definition for a View
    """
    def __init__(self, img_path:str, mask_path:str, P:npt.NDArray, dist):
        self.img = np.array(cv2.imread(img_path))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.mask = np.array(cv2.imread(mask_path))
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask[self.mask > 0] = 1
        self.mask = self.mask.astype(bool)
        self.proj_matrix = P
        self.dist = dist

        self.dst_bg = get_distance_map(self.mask)
        self.dst_fg = get_distance_map(~self.mask)

    def get_proj(self):
        return self.proj_matrix

    def get_img(self):
        return self.img

    def get_mask(self):
        return self.mask

class Node:
    """
    Class definition for a Node in the octree representation
    """
    def __init__(self, bounds, depth=0, max_depth=6):
        """
        Initialize
        """
        self.bounds = bounds # xmin, xmax, ymin, ymax, zmin, zmax
        self.depth = depth
        self.max_depth = max_depth

        self.children = []
        self.parent = None
        self.state = "full"

    def subdivide(self):
        """
        Subdivides the root node into 8 children
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        xm = (xmin + xmax) / 2
        ym = (ymin + ymax) / 2
        zm = (zmin + zmax) / 2

        boxes = [
            (xmin, xm, ymin, ym, zmin, zm),
            (xm, xmax, ymin, ym, zmin, zm),
            (xmin, xm, ymin, ym, zm, zmax),
            (xm, xmax, ymin, ym, zm, zmax),
            (xmin, xm, ym, ymax, zmin, zm),
            (xm, xmax, ym, ymax, zmin, zm),
            (xmin, xm, ym, ymax, zm, zmax),
            (xm, xmax, ym, ymax, zm, zmax)
        ]

        self.children = [Node(b, depth=self.depth+1, max_depth = self.max_depth) for b in boxes]

    def get_corners(self):
        """
        Get the coordinates of the corners of a node
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return np.array([
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymax, zmin],
                [xmin, ymax, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax]
        ])

def compute_bounds(views, z_min=0.1, z_max=10.0):
    points_3d = []
    for view in views:
        mask = view.get_mask()
        P = view.get_proj()
        P_inv = np.linalg.pinv(P)

        y_idx, x_idx = np.where(mask > 0)
        if len(x_idx) == 0:
            continue

        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)

        corners_2d = np.array([
            [x_min, y_min, 1], [x_max, y_min, 1],
            [x_min, y_max, 1], [x_max, y_max, 1]
        ])

        for c in corners_2d:
            p_near = P_inv @ (c * z_min)
            p_far = P_inv @ (c * z_max)

            points_3d.append(unhomogenize(p_near))
            points_3d.append(unhomogenize(p_far))

    points_3d = np.array(points_3d)
    min_bound = np.min(points_3d, axis=0)
    max_bound = np.max(points_3d, axis=0)

    return min_bound, max_bound

def cubify(min_bound, max_bound):
    ctr = (min_bound + max_bound) / 2
    max_side = np.max(max_bound - min_bound)
    half_side = max_side / 2

    new_min = ctr - half_side
    new_max = ctr + half_side

    return new_min, new_max

def project_points(P, X):
    """
    Projects a 3d point X to 2d point x using P
    """
    return unhomogenize(P @ homogenize(X).T)

def get_distance_map(img):
    h, w = img.shape
    dst = img.copy()
    for i in range(h):
        for j in range(w-1, -1, -1):
            if img[i, j] > 0:
                if i == 0 or j == w-1:
                    dst[i, j] = 1
                else:
                    dst[i, j] = 1 + min(dst[i-1, j], dst[i, j+1], dst[i-1, j+1])
    return dst

def classify_node(node, views):
    # print(f"Classifying node at depth {node.depth} with {len(node.children)} children...")
    all_inside = True
    for view in views:
        # get mask
        mask = view.get_mask()
        dst_bg = view.dst_bg
        dst_fg = view.dst_fg

        # get corners
        corners = node.get_corners()
        P = view.get_proj()
        x = project_points(P, corners).T
        if x is None:
            return "empty"
        
        x_min, x_max = int(np.min(x[0, :])), int(np.max(x[0, :]))
        y_min, y_max = int(np.min(x[1, :])), int(np.max(x[1, :]))
        h, w = mask.shape

        # initial check for exclusion -- bounds
        out_conds_bounds = [x_max < 0, y_max < 0, x_min >= w, y_min >= h]
        if any(out_conds_bounds):
            return "empty"
        
        x_centre, y_centre = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        # ensure centre is within img bounds
        x_centre, y_centre = np.clip(x_centre, 0, w-1), np.clip(y_centre, 0, h-1)
        
        # compute Euclidean dist from centre to every pt in bbox
        r = np.sqrt(((x_max - x_min)/2)**2 + ((y_max - y_min)/2)**2)

        if dst_fg[y_centre, x_centre] > r:
            # complete outside
            return "empty"
        
        if dst_bg[y_centre, x_centre] <= r:
            # not completely inside for this view
            all_inside = False

    if all_inside:
        # projected cubes are inside silhouette for all views
        return "full"
    else:
        # ambiguous
        return "unknown"
    

def carve_voxels(node, views):
    state = classify_node(node, views)

    if state == "empty":
        node.state = "empty"
        return

    if state == "full":
        node.state = "full"
        return

    if node.depth >= node.max_depth:
        node.state = "full"
        return

    node.subdivide()

    for child in node.children:
        carve_voxels(child, views)

def homogenize(arr):
    """
    Converts Euclidean coordinates to homogeneous coordinates

    Args:
        arr (NDArray, ((1,) or (n, m)): vector or array of m-vectors
            to homogenize
    Returns:
        NDArray ((1,) or (n, m)): homogenized vector or array of 
            m-vectors
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if len(arr.shape) < 2:
        # one vector
        return np.append(arr, 1)
    else:
        ones = np.ones((arr.shape[0],))
        return np.hstack((arr, ones[:, np.newaxis]))

def unhomogenize(vec: npt.NDArray):
    """
    Unhomogenizes a vector
    """
    n = vec.shape[0] - 1
    return vec[:n] / vec[n]

def get_vid_frames(vid_path, output_path):
    cap = cv2.VideoCapture(vid_path)
    frame_ctr = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Read failed")
            break

        cv2.imwrite(
                os.path.join(output_path, f"frame{frame_ctr:05d}.png"),
                frame)

        frame_ctr += 1

    return

def compute_proj_matrix(camera_idx, decomp=False):
    K = np.load("media/calibration/instrinsic_matrix.npy") # 3x3
    dist = np.load("media/calibration/distortion_coeffs.npy") # 1x5
    R = np.load("media/calibration/rot_vecs.npy") # n_views x 3 x 1
    t = np.load("media/calibration/trans_vecs.npy") # n_views x 3 x 1

    R, _ = cv2.Rodrigues(R[camera_idx])
    rt = np.hstack((R, t[camera_idx]))
    if not decomp:
        return dist, np.dot(K, rt)
    else:
        return K, dist, R, t


def export_to_ply(node, filename):
    points = []
    def collect_points(node):
        if not node.children:
            if node.state == "full":
                b = node.bounds
                ctr = [(b[0] + b[1])/2,
                       (b[2] + b[3])/2,
                       (b[4] + b[5])/2]
                points.append(ctr)
        else:
            for child in node.children:
                if not child is None:
                    collect_points(child)
    collect_points(node)
    points = np.array(points)
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt='%f %f %f')
    print(f"Saved {len(points)} voxels to {filename}")

def test_dst_map():
    test_img1 = np.array([
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 0]
    ])

    test_img2 = np.array([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0]
    ])

    truth_img1 = np.array([
        [0, 1, 1],
        [1, 1, 0],
        [2, 1, 0]
    ])

    truth_img2 = np.array([
        [1, 1, 1, 1],
        [0, 2, 2, 1],
        [0, 3, 2, 1],
        [0, 2, 1, 0]
    ])

    res_img1 = get_distance_map(test_img1)
    res_img2 = get_distance_map(test_img2)

    print("Image 1:")
    print(res_img1)
    print(truth_img1)

    print("Image 2:")
    print(res_img2)
    print(truth_img2)