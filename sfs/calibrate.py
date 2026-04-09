import cv2
import numpy as np
import glob
import os

def get_calib_info(filepath):
    projs = dict()
    for i, fname in enumerate(sorted(os.listdir(filepath))):
        P = np.loadtxt(os.path.join(filepath, fname), skiprows=1, dtype=np.float32)
        view_name = f"view{i:05d}"
        K, R, t, dist = decompose(P)
        projs[view_name] = {
            "P": P,
            "K": K,
            "R": R,
            "t": t,
            "dist": dist
        }
    np.savez("media/bunny_data/projs.npz", **projs)

def decompose(P):
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T.reshape(-1)
    C = T[:3] / T[3]
    t = -R @ C
    return K, R, t, None

def main():
    ## Code adapted from OpenCV documentation
    ## https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob("media/calibration/calibration_frames/*.png")
    n_views = len(images)

    # initialize values
    retval = float("inf")
    mtx = np.zeros((3, 3))
    dist = np.zeros((5,))
    rvecs = np.zeros((n_views, 3, 1))
    tvecs = np.zeros((n_views, 3, 1))
    data = {}

    for fname in images:
        img = np.array(cv2.imread(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey(500)

        retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                None,
                None)

    # evaluate
    print(f"Root mean square reprojection error: {retval:.4f}")

    # save values
    for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i]

        data[f"view{i:05d}"] = {
                "K": mtx,
                "dist": dist,
                "R": R,
                "t": t
            }

    np.savez("media/testing/test_real_data/projs", **data)

    cv2.destroyAllWindows()

    # calibrate for the bunny data
    get_calib_info("media/bunny_data/calib")

if __name__ == "__main__":
    main()
