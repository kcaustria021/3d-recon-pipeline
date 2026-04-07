import numpy as np
from numpy import typing as npt
import cv2
import os

"""
CLASSES
"""
class View:
    """
    Class definition for a View
    """
    def __init__(self, img_path:str, mask_path:str, proj_props:npt.NDArray):
        self.img = np.array(cv2.imread(img_path))

        self.mask = np.array(cv2.imread(mask_path))
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        self.mask[self.mask > 0] = 1
        self.mask = self.mask.astype(bool)

        self.contours = extract_contours(self.mask)

        self.tangents = get_tangent_lines(self.contours)

        self.dst_bg = get_distance_map((self.mask).astype(np.uint8))
        self.dst_fg = get_distance_map((~self.mask).astype(np.uint8))

        # compute projection matrix
        self.proj_props = proj_props
        try:
            self.K = self.proj_props.get("K")
            self.R = self.proj_props.get("R")
            self.t = self.proj_props.get("t").reshape(3, 1)
            self.dist = self.proj_props.get("dist")
            rt = np.hstack((self.R, self.t))
            self.P = self.K @ rt
            _, _, Vt = np.linalg.svd(self.P)
            self.cam_centre = Vt[-1, :3] / Vt[-1, 3]
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
    if len(vec.shape) == 1:
        n = vec.shape[0] - 1
        return vec[:n] / vec[n]
    elif len(vec.shape) == 2:
        n = vec.shape[1] - 1
        return vec[:, :n] / np.clip(np.expand_dims(vec[:, n], axis=1), 1e-6, None)

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
    return unhomogenize(result.T)

def get_fundamental_matrices(views):
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
    img= img.astype(np.uint8)
    dst = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
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

def get_point_status(Xs, views, eps_init=2.0):
    eps = eps_init * len(views) / 10
    n_pts = Xs.shape[0]
    status = np.full(n_pts, 2, dtype=int)

    for view in views:
        P = view.get_proj()
        dst_map = view.dst_bg
        mask = view.mask
        h, w = dst_map.shape

        X_proj = project_points(P, Xs)
        u, v = X_proj[:, 0], X_proj[:, 1]

        # check out of bounds
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        status[~in_img] = 0
        
        # check remaining points
        idx = np.where(status > 0)[0]
        u_int = np.floor(u[idx]).astype(np.int32)
        v_int = np.floor(v[idx]).astype(np.int32)
        
        m_vals = mask[v_int, u_int]
        d_vals = dst_map[v_int, u_int]

        # completely outside for one view
        is_outside = (m_vals == 0)
        status[idx[is_outside]] = 0

        # Mark points near the boundary as "pending" (1) if they weren't already marked outside (0)
        is_pending = (m_vals > 0) & (d_vals <= eps)
        # Only update if it wasn't already set to 0
        pending_indices = idx[is_pending]
        status[pending_indices] = np.minimum(status[pending_indices], 1)

    return status

def interpolate(X, direction, views, status, n_iters=5):
    low, high = 0.0, 1.0
    X_ref = X
    for _ in range(n_iters):
        mid = (low + high) / 2
        X_test = X + direction * mid
        test_status = get_point_status(X_test[np.newaxis], views)[0]
        if test_status == 0:
            high = mid
        else:
            low = mid
            X_ref = X_test
    return X_ref

def extract_contours(mask):
    """
    Get contours from the binary masks
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0].squeeze()

def get_tangent_lines(contours):
    tangents = []
    for i in range(contours.shape[0]):
        x, y = contours[i]
        pt_prev = contours[(i-1) % len(contours)]
        pt_next = contours[(i+1) % len(contours)]

        t_i = pt_next - pt_prev
        t_i_norm = np.linalg.norm(t_i)
        a, b = t_i / t_i_norm
        c = -a * x - b * y
        ell = np.array([a, b, c], dtype=np.float32)
        tangents.append(ell)
    return np.array(tangents)

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

def reconstruct(views, Fs):
    surface_points = []
    for i in range(len(views)):
        view_curr = views[i]
        view_prev = views[(i-2) % len(views)]
        view_next = views[(i+2) % len(views)]

        P_curr = view_curr.P
        P_prev = view_prev.P
        P_next = view_next.P

        contour_curr = view_curr.contours
        contour_prev = view_prev.contours
        contour_next = view_next.contours

        tangents_curr = view_curr.tangents
        tangents_prev = view_prev.tangents
        tangents_next = view_next.tangents

        cam_centre_curr = view_curr.cam_centre
        cam_centre_prev = view_prev.cam_centre
        cam_centre_next = view_next.cam_centre

        valid_pts = []
        for j in range(contour_curr.shape[0]):
            pt_prev, idx_prev = symmetric_match(
                    Fs[i][(i-2) % len(views)],
                    Fs[(i-2) % len(views)][i],
                    contour_curr[j],
                    contour_prev,
                    contour_curr)
            pt_next, idx_next = symmetric_match(
                    Fs[i][(i+2) % len(views)],
                    Fs[(i+2) % len(views)][i],
                    contour_curr[j],
                    contour_next,
                    contour_curr)

            if pt_prev is None or pt_next is None or idx_prev is None or idx_next is None:
                # null
                continue

            # current view
            ray_curr = get_visual_ray(P_curr, contour_curr[j], cam_centre_curr)
            planes_curr = backproject_tangent_plane(
                P_curr, tangents_curr, contour_curr, j
            )
            contributions_curr = [(p, ray_curr, cam_centre_curr) for p in planes_curr]
        
            # prev view
            ray_prev = get_visual_ray(P_prev, contour_prev[idx_prev], cam_centre_prev)
            planes_prev = backproject_tangent_plane(
                P_prev, tangents_prev, contour_prev, idx_prev
            )
            contributions_prev = [(p, ray_prev, cam_centre_prev) for p in planes_prev]
        
            # next view
            ray_next = get_visual_ray(P_next, contour_next[idx_next], cam_centre_next)
            planes_next = backproject_tangent_plane(
                P_next, tangents_next, contour_next, idx_next
            )
            contributions_next = [(p, ray_next, cam_centre_next) for p in planes_next]
        
            all_contributions = contributions_curr + contributions_prev + contributions_next
            pt =compute_weighted_tangent(all_contributions, ray_curr, cam_centre_curr)
            if pt is None:
                continue
            valid_pts.append(pt)

        valid_pts = np.array(valid_pts)
        statuses = get_point_status(valid_pts, views)

        mask_curr = view_curr.mask
        h, w = mask_curr.shape
        for pt in valid_pts:
            is_valid = True
            for view in views:
                mask = view.mask
                h, w = mask.shape
                pt_proj = project_points(view.P, pt)
                u, v = pt_proj
                if u < 0 or u >= w or v < 0 or v >= h:
                    is_valid = False
                    break

                if mask[int(v), int(u)] == 0:
                    is_valid = False
                    break
            if is_valid:
                surface_points.append(pt)
    return np.array(surface_points)
            
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


"""
TESTING
"""

def to_gray(img):
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:] / 255.0
        background = np.ones_like(bgr, dtype=np.uint8) * 255
        img = (bgr * alpha + background * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_image(name, img):
    if img.shape[2] == 4:
        # split into BGR and alpha
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:] / 255.0
        # composite against white background
        background = np.ones_like(bgr, dtype=np.uint8) * 255
        img = (bgr * alpha + background * (1 - alpha)).astype(np.uint8)
    cv2.imshow(name, img)

def get_display_image(view):
    img = view.img
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:] / 255.0
        background = np.ones_like(bgr, dtype=np.uint8) * 255
        return (bgr * alpha + background * (1 - alpha)).astype(np.uint8)
    return img.astype(np.uint8)

def test_unhomogenize():
    arr1 = np.array([-3, -2, -1, 1])
    arr2 = np.array([-3, 2, 1])
    arr3 = np.array([
        [-3, -2, -1, 1],
        [-4, 5, 6, 1]
    ])

    print("arr1: ", arr1)
    print("arr1 unhomo: ", unhomogenize(arr1))

    print("arr2: ", arr2)
    print("arr2 unhomo: ", unhomogenize(arr2))

    print("arr3: ", arr3)
    print("arr3 unhomo: ", unhomogenize(arr3))

def verify_fundamental_matrices(Fs, views):
    for i in range(len(views)):
        for j in range(len(views)):
            if i == j:
                continue
            F = Fs[i][j]
            contour_i = views[i].contours
            contour_j = views[j].contours
            
            for k in range(0, len(contour_i), len(contour_i) // 10):
                pt_i = homogenize(contour_i[k, :2])
                line = F @ pt_i
                pt_j, _ = find_epipolar_match(line, contour_j)
                pt_j_h = homogenize(pt_j[:2])
                residual = pt_j_h @ F @ pt_i
                print(f"F[{i}][{j}] residual at point {k}: {residual:.6f}")

def visualize_matches(views, Fs, i, j, step=20):
    img_i = get_display_image(views[i])
    img_j = get_display_image(views[j])
    contour_i = views[i].contours
    contour_j = views[j].contours
    w = img_i.shape[1]

    combined = np.hstack([img_i, img_j])

    n_matches = 0
    for k in range(0, len(contour_i), step):
        pt, idx = symmetric_match(
            Fs[i][j], Fs[j][i],
            contour_i[k], contour_j, contour_i
        )
        if pt is None:
            continue
        dy = abs(int(contour_i[k, 1]) - int(pt[1]))
        print(f"dy = {dy}")
        n_matches += 1

        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt_i = tuple(contour_i[k, :2].astype(int))
        pt_j = tuple((pt[:2] + np.array([w, 0])).astype(int))

        cv2.circle(combined, pt_i, 4, color, -1)
        cv2.circle(combined, pt_j, 4, color, -1)
        cv2.line(combined, pt_i, pt_j, color, 1)

    print(f"Found {n_matches} matches")
    cv2.imshow(f"matches {i} -> {j}", combined)
    cv2.waitKey(0)

def visualize_planes(views, Fs, i):
    for k in range(0, len(views[0].contours), 20):
        pt_prev, idx_prev = symmetric_match(
            Fs[0][len(views)-1], Fs[len(views)-1][0],
            views[0].contours[k], views[len(views)-1].contours, views[0].contours
        )
        pt_next, idx_next = symmetric_match(
            Fs[0][1], Fs[1][0],
            views[0].contours[k], views[1].contours, views[0].contours
        )
        if pt_prev is None or pt_next is None or idx_prev is None or idx_next is None:
            continue
        planes_prev = backproject_tangent_plane(
            views[len(views)-1].P,
            views[len(views)-1].tangents,
            views[len(views)-1].contours, idx_prev)
        planes_next = backproject_tangent_plane(views[1].P, views[1].tangents, views[1].contours, idx_next)
        print(f"Point {k}:")
        print(f"  prev normal: {planes_prev[2][:3] / np.linalg.norm(planes_prev[2][:3])}")
        print(f"  next normal: {planes_next[2][:3] / np.linalg.norm(planes_next[2][:3])}")
