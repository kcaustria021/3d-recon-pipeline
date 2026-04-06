import numpy as np
from numpy import typing as npt
import cv2
import os
from scipy.interpolate import interp1d

"""
CLASSES
"""
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

        # compute projection matrix
        try:
            self.K = self.proj_props.get("K")
            self.R = self.proj_props.get("R")
            self.t = np.expand_dims(self.proj_props.get("t"), axis=1)
            self.dist = self.proj_props.get("dist")

            rt = np.hstack((self.R, self.t))
            self.P = self.K @ rt
        except Exception as e:
            print(f"Computing projection matrix failed: {e}")

    def get_proj(self, decomp=False):
        if not decomp:
            return self.P
        else:
            return self.K, self.R, self.t, self.dist

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

"""
GENERAL
"""

def skew(v):
    if v.ndim > 1:
        v = np.squeeze(v, axis=1)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

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

def project_points(P, X):
    """
    Projects a 3d point X to 2d point x using P
    """
    result = P @ homogenize(X).T
    return unhomogenize(result)

def get_fundamental_matrices(views):
    Fs = dict()
    for i, view_i in enumerate(views):
        Fs[i] = dict()
        K_i, R_i, t_i, _ = view_i.get_proj(decomp=True)
        K_i_inv = np.linalg.inv(K_i)
        for j, view_j in enumerate(views):
            if i == j:
                continue
            K_j, R_j, t_j, _ = view_j.get_proj(decomp=True)
            K_j_inv = np.linalg.inv(K_j)
            R_ij = R_j @ R_i.T
            t_ij = t_j - R_ij @ t_i
            Fs[i][j] = K_j_inv.T @ skew(t_ij) @ R_ij @ K_i_inv
    return Fs

"""
VANILLA SFS
"""

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

def get_distance_map(img):
    # cv2.distanceTransform requires uint8 input
    img_uint8 = img.astype(np.uint8)
    dst = cv2.distanceTransform(img_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dst

def classify_node(node, views):
    all_inside = True
    for view in views:
        mask = view.get_mask()
        P = view.get_proj()
        corners = node.get_corners()
        centre = node.get_centre()
        dst_bg = view.dst_bg
        dst_fg = view.dst_fg
        h, w = dst_fg.shape

        corners_proj = project_points(P, corners).T  # 8 x 2
        centre_proj = project_points(P, centre.reshape(1, 3)).T  # 1 x 2

        if corners_proj is None or centre_proj is None:
            return "empty"

        u_centre, v_centre = centre_proj[0]
        u_centre_i = int(np.round(u_centre))
        v_centre_i = int(np.round(v_centre))

        # Compute bounding radius from corners to centre in projection
        u_sq_diffs = np.square(corners_proj[:, 0] - u_centre)
        v_sq_diffs = np.square(corners_proj[:, 1] - v_centre)
        r = float(np.max(np.sqrt(u_sq_diffs + v_sq_diffs)))

        # --- FIX 1: Don't reject just because centre is outside image ---
        # Instead, clamp the centre to image bounds for distance lookups,
        # or use corner projections directly.
        centre_in_image = (
            0 <= u_centre_i < w and 0 <= v_centre_i < h
        )

        if centre_in_image:
            # Standard check: is the node completely outside the foreground?
            if dst_fg[v_centre_i, u_centre_i] > r:
                return "empty"
            # Is the node completely inside for this view?
            if dst_bg[v_centre_i, u_centre_i] <= r:
                all_inside = False  # boundary node — not fully inside
        else:
            # --- FIX 2: Centre outside image; check corners individually ---
            # If ALL corners project outside the image, node is empty
            u_corners = corners_proj[:, 0]
            v_corners = corners_proj[:, 1]
            any_corner_in_image = np.any(
                (u_corners >= 0) & (u_corners < w) &
                (v_corners >= 0) & (v_corners < h)
            )
            if not any_corner_in_image:
                return "empty"

            # Check each corner that lands in-image against distance transforms
            corner_statuses = []
            for u_c, v_c in zip(u_corners, v_corners):
                u_ci, v_ci = int(np.round(u_c)), int(np.round(v_c))
                if 0 <= u_ci < w and 0 <= v_ci < h:
                    # Is this corner inside the silhouette?
                    corner_statuses.append(mask[v_ci, u_ci] > 0)

            if len(corner_statuses) == 0 or not any(corner_statuses):
                return "empty"
            # Mixed or partially inside → treat as unknown boundary
            all_inside = False

    if all_inside:
        return "full"
    else:
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

"""
HALF-SPACE SFS
"""

def extract_contours(mask):
    """
    Get contours from the binary masks
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0].squeeze()

def get_epipolar_line(F, pt):
    """
    Compute the epipolar line at a point
    """
    x = homogenize(pt).reshape((-1, 1))
    return F @ x

def find_epipolar_match(line, contour):
    """
    Find the point in another image corresponding to the
    epipolar constraint
    """
    a, b, c = line
    dists = np.abs(a * contour[:, 0] + b * contour[:, 1] + c) / np.sqrt(a**2 + b**2)
    min_idx = np.argmin(dists)
    return contour[min_idx]

def symmetric_match(F_ij, F_ji, pt_i, contour_j, contour_i):
    """
    Determine whether a match x_i -> x_k also has match x_k -> x_i
    within some error bound
    """
    pt_j = find_epipolar_match(get_epipolar_line(F_ij, pt_i), contour_j)
    pt_i_back = find_epipolar_match(get_epipolar_line(F_ji, pt_j), contour_i)
    return pt_j if np.linalg.norm(pt_i - pt_i_back) < 2.0 else None

def backproject_ray(P, point):
    """
    Back-project a point to a ray
    """
    x = homogenize(point)
    ray = np.linalg.pinv(P) @ x
    return unhomogenize(ray)

def triangulate(P1, P2, pt1, pt2):
    """
    Triangulate a 3d point using projection matrices and 2d points
    """
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return unhomogenize(X)

def inside_silhouette(pt, mask):
    """
    Check whether a point is inside the object in a binary mask
    """
    x, y = int(round(pt[0])), int(round(pt[1]))
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return mask[y, x] > 0

def reconstruct(P_list, silhouettes, contours, F_matrices):
    """
    Perform the reconstruction
    """
    surface_points = []
    for i, contour_i in enumerate(contours):
        for pt_i in contour_i:
            valid_points = []
            for k in range(len(contours)):
                if k == i:
                    continue
                pt_k = symmetric_match(
                    F_matrices[i][k],
                    F_matrices[k][i],
                    pt_i,
                    contours[k],
                    contour_i
                )
                if pt_k is not None:
                    # guess a 3d point
                    X = triangulate(P_list[i], P_list[k], pt_i, pt_k)
                    valid_points.append(X)

            if len(valid_points) >= 2:
                X_avg = np.mean(valid_points, axis=0)
                good_pt = True
                for cam, mask in zip(P_list, silhouettes):
                    proj = project_points(cam, X_avg)
                    if not inside_silhouette(proj, mask):
                        good_pt = False
                        break
                if good_pt:
                    surface_points.append(X_avg)

    return np.array(surface_points)

def interpolate_points(points):
    known = np.array([p for p in points if p is not None])
    return known

"""
I/O
"""

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
    with open(f"{filename}.ply", "w") as f:
        f.write(header)
        try:
            np.savetxt(f, points, fmt='%f %f %f')
        except Exception as e:
            print(e)
    print(f"Saved {len(points)} voxels to {filename}")
