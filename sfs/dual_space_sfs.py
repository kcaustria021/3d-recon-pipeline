# pylint: disable=consider-using-enumerate

from typing import List
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from numpy import typing as npt

from vol_sfs import get_distance_map, symmetric_match, project_points

class View2:
    def __init__(self, img_path:str, mask_path:str, proj_props:npt.NDArray):
        self.img = np.array(cv2.imread(img_path))

        if mask_path.endswith(".png"):
            self.mask = np.array(cv2.imread(mask_path))
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
            self.mask[self.mask > 0] = 1
            self.mask = self.mask.astype(bool)
        elif mask_path.endswith(".pgm"):
            self.mask = np.array(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED))
            self.mask = ~(self.mask.astype(bool))

        # Handle multiple contours
        self.contours = extract_contours_multi(self.mask)
        
        self.splines_tck = []
        self.splines_u = []
        for c in self.contours:
            tck, u = fit_bspline(c)
            self.splines_tck.append(tck)
            self.splines_u.append(u)

        # (Keeping standard distance map logic)
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

def extract_contours_multi(mask):
    """
    Get all contours (e.g., torus outer edge and inner hole).
    Returns a list of contour arrays.
    """
    mask = mask.astype(np.uint8)
    # Using RETR_LIST gets all contours regardless of nesting
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Filter out tiny artifacts and squeeze
    valid_contours = []
    for c in contours:
        c = c.squeeze()
        if c.ndim == 2 and c.shape[0] >= 10: 
            valid_contours.append(c)
    return valid_contours

def fit_bspline(contours, smooth=0.0):
    points = [contours[:, 0], contours[:, 1]]
    tck, u = splprep(points, s=smooth, k=3, per=True)
    return tck, u

def get_tangent_plane_analytic(P, tck, s):
    """
    Helper to cleanly evaluate the plane at spline parameter s
    """
    x, y = splev(s, tck)
    dx, dy = splev(s, tck, der=1)
    
    norm = np.hypot(dx, dy) + 1e-8
    nx, ny = -dy / norm, dx / norm # normal direction (unit)
    
    c = -(nx * float(x) + ny * float(y))
    l_2d = np.array([nx, ny, c], dtype=np.float32) # get 2d tangent line to s
    
    plane = P.T @ l_2d # back project to tgt plane
    return plane / np.linalg.norm(plane[:3])

def get_3_plane_set(views: List[View2], Fs, contour_idx: int, s_idx: int, t: int):
    """
    Returns the 3 planes list [r~*(s, t-1), r~*(s, t), r~*(s, t+1)]
    Matches purely via index mapped to the spline `u` parameter.
    """
    view_curr = views[t]
    view_prev = views[(t-1) % len(views)]
    view_next = views[(t+1) % len(views)]
    
    # Isolate the specific contour we are working on
    contour_curr = view_curr.contours[contour_idx]
    contour_prev = view_prev.contours[contour_idx]
    contour_next = view_next.contours[contour_idx]
    
    tck_curr = view_curr.splines_tck[contour_idx]
    u_curr = view_curr.splines_u[contour_idx]
    
    pt_curr = contour_curr[s_idx]
    s_curr = u_curr[s_idx]
    
    # 1. Plane at t
    plane_curr = get_tangent_plane_analytic(view_curr.P, tck_curr, s_curr)
    
    # epipolar match at t-1
    pt_prev, idx_prev = symmetric_match(
                    Fs[t][(t-1) % len(views)],
                    Fs[(t-1) % len(views)][t],
                    pt_curr,
                    contour_prev,
                    contour_curr)
    
    # epipolar match at t+1
    pt_next, idx_next = symmetric_match(
            Fs[t][(t+1) % len(views)],
            Fs[(t+1) % len(views)][t],
            pt_curr,
            contour_next,
            contour_curr)
    
    if pt_prev is None or pt_next is None or idx_prev is None or idx_next is None:
        return None

    # Use the untouched symmetric match indices to fetch exact 's' parameters
    s_prev = view_prev.splines_u[contour_idx][idx_prev]
    s_next = view_next.splines_u[contour_idx][idx_next]
    
    plane_prev = get_tangent_plane_analytic(view_prev.P, view_prev.splines_tck[contour_idx], s_prev)
    plane_next = get_tangent_plane_analytic(view_next.P, view_next.splines_tck[contour_idx], s_next)

    return [plane_prev, plane_curr, plane_next]

def get_gaussian_weight(ds, dt, sigma_s=2.0, sigma_t=1.0):
    """
    Computes a 2D Gaussian penalty over the contour step (ds) 
    and the time step (dt).
    """
    return np.exp(-(ds**2 / (2 * sigma_s**2)) - (dt**2 / (2 * sigma_t**2)))


def get_ray_direction(view, u, v):
    """
    Computes the normalized 3D ray direction vector from the camera center 
    through the 2D pixel coordinate.
    """
    K_inv = np.linalg.inv(view.K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = view.R.T @ ray_cam
    return ray_world / np.linalg.norm(ray_world)

def get_visual_ray(P, pt, cam_centre):
    pt_h = np.array([pt[0], pt[1], 1.0], dtype=np.float32) # homogenous point
    ray = np.linalg.pinv(P) @ pt_h # back project point and produce a world point
    ray = ray[:3] / ray[3] - cam_centre # world point displacement from camera centre
    return ray / np.linalg.norm(ray)

def backproject_tangent_plane_analytic(P, splines_tck, splines_u, contour_idx, s_idx, n_pts, window_s=2):
    """
    Analytically maps the integer neighborhood indices back to the smooth B-splines
    to generate mathematically exact local tangent envelopes.
    """
    planes = []
    tck = splines_tck[contour_idx]
    u = splines_u[contour_idx]
    
    for i in range(-window_s, window_s + 1):
        j = (s_idx + i) % n_pts
        s = u[j]
        plane = get_tangent_plane_analytic(P, tck, s)
        planes.append(plane)
        
    return planes

def compute_weighted_tangent(contributions, ray_curr, cam_centre_curr):
    consistent = []
    for p, r, c in contributions:
        # normalize planes
        p_normalized = p / np.linalg.norm(p[:3])
        # consistent orientation: camera outside the plane
        if np.dot(p_normalized[:3], ray_curr) < 0:
            p_normalized = -p_normalized
        consistent.append((p_normalized, r, c))

    # 1 / |n * r| ie weight by perpendicularity of ray to the tangent plane, it decays quadratically
    weights = np.array([np.abs(np.dot(p[:3] / p[3], r)) for p, r, c in consistent])
    weights = 1.0 / (weights + 1e-8)
    weights /= weights.sum()

    # there must be a better way than to use the average plane 
    avg_plane = sum(w * p for w, (p, r, c) in zip(weights, consistent))
    n, d = avg_plane[:3], avg_plane[3]

    # solve for depth, X = C + t * R, nX + d = 0 solve for t.
    denom = n @ ray_curr
    t = -(n @ cam_centre_curr + d) / denom

    # skip degenerate endpoints
    if np.abs(denom) < 1e-2 or t < 0:
        return None

    return cam_centre_curr + t * ray_curr

def reconstruct2(views, Fs, window_s=2):
    """
    1. get 3-plane set {r~∗(s,t−1),r~∗(s,t),r~∗(s,t+1)} for each control point s
    2. get the same thing for neighbors (s-e, s+e)
    3. compute weights for each of these guys with a gaussian
    4. large weight on principal plane r~∗(s,t) and on the visual ray constraint (O(t) x w(s, t)) * X = 0
    5. stack all of them into a matrix and put MX = 0, solve with SVD or regression (fix W to 1, may not work at POI)
    6. that will give you X, now repeat for all control points.
    """
    surface_points = []
    
    for t in range(len(views)):
        view_curr = views[t]
        view_prev = views[(t-1) % len(views)]
        view_next = views[(t+1) % len(views)]

        P_curr = view_curr.P
        P_prev = view_prev.P
        P_next = view_next.P

        cam_centre_curr = view_curr.cam_centre
        cam_centre_prev = view_prev.cam_centre
        cam_centre_next = view_next.cam_centre

        valid_pts = []
        
        # 1. Loop through all independent contours
        for contour_idx in range(len(view_curr.contours)): 
            print(f"View {t}, Contour {contour_idx} of {len(view_curr.contours)}")
            
            # Ensure neighboring views actually saw the same number of boundaries
            if contour_idx >= len(view_prev.contours) or contour_idx >= len(view_next.contours):
                continue
                
            contour_curr = view_curr.contours[contour_idx]
            contour_prev = view_prev.contours[contour_idx]
            contour_next = view_next.contours[contour_idx]
            
            n_pts_curr = len(contour_curr)
            n_pts_prev = len(contour_prev)
            n_pts_next = len(contour_next)

            # 2. Iterate points exactly along the continuous spline
            for j in range(n_pts_curr):
                pt_prev, idx_prev = symmetric_match(
                        Fs[t][(t-1) % len(views)],
                        Fs[(t-1) % len(views)][t],
                        contour_curr[j], contour_prev, contour_curr)
                        
                pt_next, idx_next = symmetric_match(
                        Fs[t][(t+1) % len(views)],
                        Fs[(t+1) % len(views)][t],
                        contour_curr[j], contour_next, contour_curr)

                if pt_prev is None or pt_next is None or idx_prev is None or idx_next is None:
                    continue

                # ray & plane triple, to avoid camera centre singularity with SVD
                ray_curr = get_visual_ray(P_curr, contour_curr[j], cam_centre_curr)
                ray_prev = get_visual_ray(P_prev, contour_prev[idx_prev], cam_centre_prev)
                ray_next = get_visual_ray(P_next, contour_next[idx_next], cam_centre_next)
                
                planes_curr = backproject_tangent_plane_analytic(
                    P_curr, view_curr.splines_tck, view_curr.splines_u, contour_idx, j, n_pts_curr
                )
                planes_prev = backproject_tangent_plane_analytic(
                    P_prev, view_prev.splines_tck, view_prev.splines_u, contour_idx, idx_prev, n_pts_prev
                )
                planes_next = backproject_tangent_plane_analytic(
                    P_next, view_next.splines_tck, view_next.splines_u, contour_idx, idx_next, n_pts_next
                )

                contributions_curr = [(p, ray_curr, cam_centre_curr) for p in planes_curr]
                contributions_prev = [(p, ray_prev, cam_centre_prev) for p in planes_prev]
                contributions_next = [(p, ray_next, cam_centre_next) for p in planes_next]

                # basically get a plane triple {r~∗(s,t−1),r~∗(s,t),r~∗(s,t+1)} for each control point s
            
                # kims tangent heuristic,.
                # each contribution is a plane, its corresponding ray direction from the camera centre, and the actual camera centre
                all_contributions = contributions_curr + contributions_prev + contributions_next
                # weighted tangent of the dual gives back a 3d point in primal
                pt = compute_weighted_tangent(all_contributions, ray_curr, cam_centre_curr)
                
                if pt is not None:
                    valid_pts.append(pt)

        # check if points actually project back into the mask
        for pt in valid_pts:
            is_valid = True
            for view in views:
                mask = view.mask
                h, w = mask.shape
                pt_proj = project_points(view.P, pt)
                u, v = pt_proj[0] if pt_proj.ndim == 2 else pt_proj
                
                if u < 0 or u >= w or v < 0 or v >= h:
                    is_valid = False
                    break

                if mask[int(v), int(u)] == 0:
                    is_valid = False
                    break
                    
            if is_valid:
                surface_points.append(pt)
                
    return np.array(surface_points)