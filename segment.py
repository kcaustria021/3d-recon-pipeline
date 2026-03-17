import cv2
import numpy as np
import argparse

def get_args():
    """
    Process the given arguments
    """
    parser = argparse.ArgumentParser(
            description="Argument parser for image segmentation",
            prog="segment.py")

    parser.add_argument("img_path", type=str, help="The name of the image to be segmented")
    return parser.parse_args()

def click_event(event, x, y, flags, param):
    """
    Mouse callback event for selecting points on the silhouette
    of the image

    Args:
        event (int): the event that occurred
        x (int): the x-coordinate where the event occurred
        y (int): the y-coordinate where the event occurred
        flags (int): the flags for the callback; see documentation
        param (any): any variables useful for the click event; see documentation
    Returns:
        None
    """
    points = param.get("points")
    img = param.get("img")
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, ((x, y)), radius=2, color=(0, 255, 0), thickness=2)
        cv2.imshow("Image", img)

def get_mask(img):
    """
    Obtains the binary foreground-background mask to detect the silhouette
    of the object; done manually for now

    Args:
        img (np.array): the image to get the mask from
    Returns:
        (np.array): the binary mask for the object, same shape as img
    """
    # TODO: implement a tracker to track points across images
    points = []
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event, param={"points": points, "img": img})

    points = np.array(points, np.int32)
    polygon_pts = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon_pts], color=(255, 255, 255))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask[mask == 255] = 1

    return mask

def main():
    args = get_args()
    img_path = args.img_path
    try:
        img = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR_RGB))
        mask = get_mask(img)
    except FileNotFoundError:
        print(f"File {img_path} not found, exiting...")


if __name__ == "__main__":
    main()
