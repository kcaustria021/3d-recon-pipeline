import cv2
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(
            description="Argument parser for background subtraction"
        )

    parser.add_argument("-w", "--write", action="store_true", help="To write or not to write frames of the video and masks")
    return parser.parse_args()

def write_frame(output_dir: str, frame, frame_ct: int):
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        try:
            os.rmdir(output_dir)
            print(f"Directory {output_dir} deleted.")
        except FileNotFoundError:
            print(f"Error: {output_dir} not found")
        except OSError as e:
            print(f"Error: {e}. {output_dir} is not empty.")

    os.mkdir(output_dir)
    fname = os.path.join(output_dir, f"frame{frame_ct:06d}.jpg")
    cv2.imwrite(fname, frame)

def main():
    cap = cv2.VideoCapture("media/test_bgs.mov")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    args = get_args()
    to_write = args.write

    vid_frames_dir = os.path.join("media", "vid_frames")
    mask_frames_dir = os.path.join("media", "masked_frames")

    frame_ct = 0
    print("Press Esc to close the window.")
    while True:
        ret, frame = cap.read()
    
        if ret is None:
            break
    
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        if to_write:
            write_frame(vid_frames_dir, frame, frame_ct)
            write_frame(mask_frames_dir, frame, frame_ct)
    
        cv2.imshow("frame", fgmask)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

        frame_ct += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
