import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("vid_path")
    parser.add_argument("output_path")

    args = parser.parse_args()
    vid_path = args.vid_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cap = cv2.VideoCapture(vid_path)
    frame_ctr = 0

    print("Press 's' to select frames to save")
    print("Press any other key to continue")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0) & 0xFF
        if k == 115:
            # 's' pressed
            frame_name = os.path.join(output_path, f"frame{frame_ctr:05d}.png")
            cv2.imwrite(frame_name, frame)
            frame_ctr += 1
        else:
            continue

if __name__ == "__main__":
    main()
