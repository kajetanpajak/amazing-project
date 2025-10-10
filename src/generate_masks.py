import cv2
import numpy as np
import os

def yolo_seg_to_mask(img_path, label_path, output_path):
    # Load image to get dimensions
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # Create blank mask
    mask = np.zeros((H, W), dtype=np.uint8)

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])  # class id
            coords = list(map(float, parts[1:]))

            # Group into (x, y) pairs
            points = np.array([
                [int(coords[i] * W), int(coords[i+1] * H)]
                for i in range(0, len(coords), 2)
            ], dtype=np.int32)

            # Fill polygon white
            cv2.fillPoly(mask, [points], 255)

    cv2.imwrite(output_path, mask)

IMAGE_FOLDER = "data/images"
YOLO_LABEL_FOLDER = "data/labels_yolo"
MASK_OUTPUT_FOLDER = "data/labels_binary"

def main():
    os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)

    for file in os.listdir(IMAGE_FOLDER):
        filename = os.path.splitext(file)[0]
        print(filename)

        yolo_label_path = f'{YOLO_LABEL_FOLDER}/{filename}.txt'
        output_path = f'{MASK_OUTPUT_FOLDER}/{filename}_mask.png'

        yolo_seg_to_mask(
            img_path=f'{IMAGE_FOLDER}/{file}',
            label_path=yolo_label_path,
            output_path=output_path
        )

if __name__ == "__main__":
    main()