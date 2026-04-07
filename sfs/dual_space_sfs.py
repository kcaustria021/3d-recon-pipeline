from typing import List
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from numpy import typing as npt

from utils_sfs import get_distance_map, symmetric_match

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

class View2:
    def __init__(self, img_path:str, mask_path:str, proj_props:npt.NDArray):
        self.img = np.array(cv2.imread(img_path))

        self.mask = np.array(cv2.imread(mask_path))
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask[self.mask > 0] = 1
        self.mask = self.mask.astype(bool)

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

def reconstruct2(views: List[View2], Fs):
    """
    # get 3-plane set {r~∗(s,t−1),r~∗(s,t),r~∗(s,t+1)} for each control point s
    # get the same thing for neighbors (s-e, s+e)
    # compute weights for each of these guys with a gaussian
    # large weight on principal plane r~∗(s,t) and on the visual ray constraint (O(t) x w(s, t)) * X = 0
    # stack all of them into a matrix and put MX = 0, solve with SVD or regression (fix W to 1, may not work at POI)
    """

    for view_idx, view in enumerate(views):
        for contour_idx, contour in enumerate(view.contours):
            for s_idx in range(len(contour)): # s is a point on the cv2 countour [x, y]
                plane_triple = get_3_plane_set(views, Fs, contour_idx, s_idx, view_idx)
                if plane_triple is not None:
                    print(plane_triple)
    return