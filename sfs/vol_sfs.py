import numpy as np
from numpy import typing as npt
import cv2
import os

"""
GENERAL
"""
def skew(v):
    """
    Skew-symmetrize a vector v
    """
    if v.ndim > 1:
        v = np.squeeze(v, axis=1)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def homogenize(arr):
    """
    Convert a Euclidean vector to homogeneous coordinates
    """
    if len(arr.shape) < 2:
        # one vector
        return np.append(arr, 1)
    else:
        # many vectors
        ones = np.ones((arr.shape[0],))
        return np.hstack((arr, ones[:, np.newaxis]))

def unhomogenize(vec: npt.NDArray):
    """
    Convert a vector in homogeneous coordinates to Euclidean
    """
    if len(vec.shape) == 1:
        n = vec.shape[0] - 1
        return vec[:n] / vec[n]
    elif len(vec.shape) == 2:
        n = vec.shape[1] - 1
        return vec[:, :n] / np.clip(np.expand_dims(vec[:, n], axis=1), 1e-6, None)

def get_vid_frames(vid_path, output_path):
    """
    Grab frames from a video
    """
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
    P is 3x4
    X is in Euclidean coordinates
    return value is a 2-vector in Euclidean coordinates

    handles projecting array of vectors
    """
    result = P @ homogenize(X).T
    return unhomogenize(result.T)

def get_fundamental_matrices(views):
    """
    Compute the fundamental matrix for the views
    """
    Fs = dict()
    for i, view_i in enumerate(views):
        Fs[i] = dict()
        K_i, R_i, t_i, _ = view_i.get_proj(decomp=True)
        C_i = -R_i.T @ t_i  # camera centre in world space
        for j, view_j in enumerate(views):
            if i == j:
                continue
            K_j, R_j, t_j, _ = view_j.get_proj(decomp=True)
            C_j = -R_j.T @ t_j

            R_ij = R_j @ R_i.T
            t_ij = R_j @ (C_i - C_j)

            E = skew(t_ij) @ R_ij
            Fs[i][j] = np.linalg.inv(K_j).T @ E @ np.linalg.inv(K_i)
            Fs[i][j] /= np.linalg.norm(Fs[i][j])
    return Fs

def compute_bounds(views, z_min=0.1, z_max=10.0):
    """
    Guess the coordinates for the bounding cube for the object of interest
    """
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
    """
    Convert normal bounds to cube bounds whose centroid
    matches the centroid of the object of interest
    """
    ctr = (min_bound + max_bound) / 2
    max_side = np.max(max_bound - min_bound)
    half_side = max_side / 2

    new_min = ctr - half_side
    new_max = ctr + half_side

    return new_min, new_max

def get_distance_map(img):
    """
    Create a distance map of the pixels' L2 distance from the nearest 0-pixel
    """
    img= img.astype(np.uint8)
    dst = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dst

def classify_node(node, views):
    """
    Determine whether a node is:
        - "empty", if it is outside the silhouette for at least one view
        - "full", if it is inside the silhouette for all views
        - "unknown", if it is inside the silhouette for some views and outside
            for other views
    """
    all_inside = True
    for view in views:
        mask = view.get_mask()
        P = view.get_proj()
        corners = node.get_corners()
        centre = node.get_centre()
        dst_bg = view.dst_bg
        dst_fg = view.dst_fg
        h, w = dst_fg.shape

        corners_proj = project_points(P, corners)  # 8 x 2
        centre_proj = project_points(P, centre.reshape(1, 3)).T  # 2 x 1

        if corners_proj is None or centre_proj is None:
            return "empty"

        u_centre, v_centre = centre_proj
        u_centre_i = int(np.round(u_centre[0]))
        v_centre_i = int(np.round(v_centre[0]))

        u_sq_diffs = np.square(corners_proj[:, 0] - u_centre)
        v_sq_diffs = np.square(corners_proj[:, 1] - v_centre)
        r = float(np.max(np.sqrt(u_sq_diffs + v_sq_diffs)))

        centre_in_image = (
            0 <= u_centre_i < w and 0 <= v_centre_i < h
        )

        if centre_in_image:
            if dst_fg[v_centre_i, u_centre_i] > r:
                return "empty"
            if dst_bg[v_centre_i, u_centre_i] <= r:
                all_inside = False
        else:
            u_corners = corners_proj[:, 0]
            v_corners = corners_proj[:, 1]
            any_corner_in_image = np.any(
                (u_corners >= 0) & (u_corners < w) &
                (v_corners >= 0) & (v_corners < h)
            )
            if not any_corner_in_image:
                return "empty"

            corner_statuses = []
            for u_c, v_c in zip(u_corners, v_corners):
                u_ci, v_ci = int(np.round(u_c)), int(np.round(v_c))
                if 0 <= u_ci < w and 0 <= v_ci < h:
                    corner_statuses.append(mask[v_ci, u_ci] > 0)

            if len(corner_statuses) == 0 or not any(corner_statuses):
                return "empty"
            all_inside = False

    if all_inside:
        return "full"
    else:
        return "unknown"

def carve_voxels(node, views):
    """
    Perform voxel carving according to:
        - remove if a node is empty
        - retain if a node is full
        - subdivide if a node is unknown
    """
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

def get_epipolar_line(F, pt):
    """
    Compute the epipolar line at a point
    """
    x = homogenize(pt).reshape((-1, 1))
    return F @ x

def find_epipolar_match(line, contour):
    """
    Find corresponding point along epipolar line
    """
    a, b, c = line
    dists = np.abs(a * contour[:, 0] + b * contour[:, 1] + c) / np.sqrt(a**2 + b**2)
    min_idx = np.argmin(dists)
    return contour[min_idx], min_idx

def symmetric_match(F_ij, F_ji, pt_i, contour_j, contour_i):
    """
    Make sure epipolar matches go both ways view i -> view k and view_k -> view_i
    """
    pt_j, idx_j = find_epipolar_match(get_epipolar_line(F_ij, pt_i), contour_j)
    pt_i_back, _ = find_epipolar_match(get_epipolar_line(F_ji, pt_j), contour_i)
    return pt_j, idx_j if np.linalg.norm(pt_i - pt_i_back) < 2.0 else None

def backproject_tangent_plane(P, tangents, pts, idx, n_pts=2):
    planes = []
    for i in range(-n_pts, n_pts+1, 1):
        j = (idx + i) % len(pts)
        tangent_line = tangents[j]
        plane = P.T @ tangent_line
        plane /= np.linalg.norm(plane[:3])
        planes.append(plane)
    return planes

def get_visual_ray(P, pt, cam_centre):
    pt_h = homogenize(pt)
    ray = np.linalg.pinv(P) @ pt_h
    ray = ray[:3] / ray[3] - cam_centre
    return ray / np.linalg.norm(ray)

def compute_weighted_tangent(contributions, ray_curr, cam_centre_curr):
    consistent = []
    for p, r, c in contributions:
        # normalize planes
        p_normalized = p / np.linalg.norm(p[:3])
        # make sure signs are consistent
        # camera outside the plane
        if np.dot(p_normalized[:3], ray_curr) < 0:
            p_normalized = -p_normalized
        consistent.append((p_normalized, r, c))

    weights = np.array([np.abs(np.dot(p[:3], r)) for p, r, c in consistent])
    weights = 1.0 / (weights + 1e-8)
    weights /= weights.sum()

    avg_plane = sum(w * p for w, (p, r, c) in zip(weights, consistent))
    n, d = avg_plane[:3], avg_plane[3]

    denom = n @ ray_curr
    t = -(n @ cam_centre_curr + d) / denom

    # skip degenerate points
    if np.abs(denom) < 1e-2 or t < 0:
        return None

    return cam_centre_curr + t * ray_curr

"""
I/O
"""

def octree_to_ply(node, filename):
    """
    Export node centres as point cloud points to ply file
    """
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

def pts_to_ply(points, ply_path):
    N = len(points)

    with open(f"{ply_path}.ply", 'w') as f:
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