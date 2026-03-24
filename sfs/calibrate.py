# TODO: implement camera calibration
# see https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import cv2
import numpy as np
import glob

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

    # save values
    print(f"Root mean square reprojection error: {retval:.4f}")
    np.save("media/calibration/instrinsic_matrix.npy", mtx)
    np.save("media/calibration/distortion_coeffs.npy", dist)
    np.save("media/calibration/rot_vecs.npy", rvecs)
    np.save("media/calibration/trans_vecs.npy", tvecs)


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
