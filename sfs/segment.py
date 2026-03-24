import cv2
import numpy as np
import argparse
import os

def get_args():
    """
    Process the given arguments
    """
    parser = argparse.ArgumentParser(
            description="Argument parser for image segmentation",
            prog="segment.py")

    parser.add_argument("img_path", type=str, help="The directory that contains images of the object")
    parser.add_argument("bg_path", type=str, help="The directory that contains images of the background")
    parser.add_argument("output_path", type=str, help="The directory that will contain the binary masks")
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

def clean_mask(mask_vis, init_mask):
    print("Clean the mask")

    x, y, w, h = cv2.selectROI("mask cleaning", mask_vis)
    x1, y1 = (int(x), int(y))
    x2, y2= (int(x + w), int(y + h))
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("selected ROI", mask_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    binary_mask = np.zeros(init_mask.shape, dtype=bool)
    binary_mask[y1:y2+1, x1:x2+1] = True

    init_mask[~binary_mask] = 0

    return init_mask

def get_mask(img, backgrounds, threshold=30):
    background = np.mean(backgrounds, axis=0).astype(np.uint8)
    diff = np.abs(img.astype(np.int16) - background.astype(np.int16))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return mask

def get_backgrounds(bg_path):
    background_frames = []
    for frame_name in sorted(os.listdir(bg_path)):
        fname = os.path.join(bg_path, frame_name) 
        if fname.endswith(".png"):
            frame = np.array(cv2.imread(os.path.join(bg_path, frame_name)), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background_frames.append(frame)
    return np.array(background_frames)

def get_frames(img_dir):
    frames = []
    for frame_name in sorted(os.listdir(img_dir)):
        fname = os.path.join(img_dir, frame_name)
        if fname.endswith(".png"):
            frame = np.array(cv2.imread(fname), dtype=np.uint8)
            frames.append(frame)
    return frames

def view_masks(mask_path):
    for mask_name in sorted(os.listdir(mask_path)):
        fname = os.path.join(mask_path, mask_name)
        if fname.endswith(".png"):
            mask = np.array(cv2.imread(fname))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            print(np.unique(mask))
            color_mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            color_mask[:, :, 2] = mask
            cv2.imshow("mask", color_mask)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    args = get_args()
    img_path = args.img_path
    bg_path = args.bg_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    frames = get_frames(img_path)
    try:
        backgrounds = get_backgrounds(bg_path)
        for i in range(len(frames)):
            print(f"Frame {i}...")
            frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            mask = get_mask(frame, backgrounds, threshold=60)
            color_mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            color_mask[:, :, 2] = mask
            cv2.imshow("mask", color_mask)
            print("Press 'y' if you want to clean the mask")
            k = cv2.waitKey(0) & 0xFF
            if k == 121:
                mask = clean_mask(color_mask, mask)
            
            write_path = os.path.join(output_path, f"mask{i:05d}.png")
            cv2.imwrite(write_path, mask)
        
        cv2.destroyAllWindows()
        view_masks(output_path)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
