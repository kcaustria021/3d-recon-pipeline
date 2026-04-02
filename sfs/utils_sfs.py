import numpy as np
from numpy import typing as npt
import cv2
import os

class View:
    """
    Class definition for a View
    """
    def __init__(self, img_path:str, mask_path:str, proj_props:npt.NDArray):
        self.img = np.array(cv2.imread(img_path))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.mask = np.array(cv2.imread(mask_path))
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        self.mask[self.mask > 0] = 1
        self.mask = self.mask.astype(bool)
        self.proj_props = proj_props

        self.dst_bg = get_distance_map((self.mask).astype(np.uint8))
        self.dst_fg = get_distance_map((~self.mask).astype(np.uint8))

    def get_proj(self, decomp=False):
        try:
            self.K = self.proj_props.get("K")
            self.R = self.proj_props.get("R")
            self.t = np.expand_dims(self.proj_props.get("t"), axis=1)
            self.dist = self.proj_props.get("dist")

            rt = np.hstack((self.R, self.t))
            self.P = self.K @ rt

            if not decomp:
                return self.P
            else:
                return self.K, self.R, self.t, self.dist
        except Exception as e:
            print(e)

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

    def get_centre(self):
        """
        Get the coordinates of the centre of the cube
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return np.array([
            (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
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
    result = P @ homogenize(X).T
    return unhomogenize(result)

def get_distance_map(img):
    # cv2.distanceTransform requires uint8 input
    img_uint8 = img.astype(np.uint8)
    dst = cv2.distanceTransform(img_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dst

def classify_node(node, views):
    # print(f"Classifying node at depth {node.depth} with {len(node.children)} children...")
    all_inside = True
    for view in views: 
        mask = view.get_mask() 
        P = view.get_proj()
        corners = node.get_corners()
        centre = node.get_centre()

        dst_bg = view.dst_bg
        dst_fg = view.dst_fg
        h, w = dst_fg.shape

        corners_proj = project_points(P, corners).T # 8 x 2
        centre_proj = project_points(P, centre.reshape(1, 3)).T # 1 x 2

        if corners_proj is None:
            return "empty"
        if centre_proj is None:
            return "empty"

        u_centre, v_centre = centre_proj[0]
        u_centre = int(np.round(u_centre))
        v_centre = int(np.round(v_centre))

        u_sq_diffs = np.square(corners_proj[:, 0] - u_centre)
        v_sq_diffs = np.square(corners_proj[:, 1] - v_centre)
        dsts = np.sqrt(u_sq_diffs + v_sq_diffs)

        if u_centre < 0 or u_centre >= w or v_centre < 0 or v_centre >= h:
            return "empty"

        r = float(np.max(dsts))

        if dst_fg[v_centre, u_centre] > r:
            # complete outside
            return "empty"

        if dst_bg[v_centre, u_centre] <= r:
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
        node.state = "empty"
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
        try:
            np.savetxt(f, points, fmt='%f %f %f')
        except Exception as e:
            print(e)
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
