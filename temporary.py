import sys
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
from collections import Counter, deque

sys.path.append(str(Path(__file__).parent / 'src'))
from datasets import get_validation_transforms
from model import SegmentationModel

from utils import *

def main():
    # directory_path = Path('E:/')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegmentationModel(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

    model_state_dict = torch.load('models/best_model.pth', map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    image_size = (512, 512)
    transform = get_validation_transforms(image_size=image_size)

    # files = os.listdir(directory_path)
    # video_names = [f for f in files if f.lower().endswith(('.mp4'))]
    videos_failed = []

    output_dir = Path('output_pendrive')
    header = ['Frame', 'Diameter (pixels)', 'Diameter (ratio)']

    directory_path = 'videos'
    video_names = ['C0209.MP4', 'C0210.MP4', 'C0211.MP4']

    for file_name in video_names:
        file_name = Path(file_name)
        file_path = os.path.join(directory_path, file_name)
       
        cap = cv.VideoCapture(file_path, cv.CAP_FFMPEG)

        if not cap.isOpened():
            print(f'Failed to open: {file_name}')
            videos_failed.append(file_name)
            continue
        else:
            print(f'Processing video: {file_path}')

        break_frame, fluid_broke = get_fluid_burst(cap, model, device, transform, False)
        end_of_expansion_frame = get_end_of_expansion(cap, model, device, transform, break_frame, fluid_broke, False)

        measurement_x = get_expected_narrowest_point_v1(cap, model, device, transform, end_of_expansion_frame,
                                                        break_frame, (0, 1080-1), False)


        # track changes of diameter
        diameters = []
        cap.set(cv.CAP_PROP_POS_FRAMES, end_of_expansion_frame - 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp_frame = frame.copy()
            disp_frame = cv.resize(disp_frame, (512, 512))
            
            mask = get_binary_mask(frame, model, device, transform)

            measurement_col = mask[:, measurement_x]
            diameter = np.sum(measurement_col)  / 255
            if diameter > 0:
                diameters.append(diameter)
                top_point = np.where(measurement_col == 255)[0][0]
            else:
                top_point = None

            if top_point:
                cv.line(disp_frame, (measurement_x, top_point), (measurement_x, top_point + int(diameter)), (0, 255, 0), 2)

            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            concatenated = np.hstack((disp_frame, mask))
            cv.imshow('Frame and Mask', concatenated)
            if cv.waitKey(20) & 0xFF == ord('q'):
                break
            if cap.get(cv.CAP_PROP_POS_FRAMES) >= break_frame:
                break

        diameter = np.array(diameters)
        diameter_ratio = diameter / diameter[0]
        frames = np.arange(len(diameter))

        csv_file_path = os.path.join(output_dir, f'{file_name.stem}_output.csv')
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for f, d, dr in zip(frames, diameter, diameter_ratio):
                writer.writerow([f, d, dr])

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(frames, diameter_ratio)
        plt.title(file_name.stem)
        plt.xlabel('Frame')
        plt.ylabel('Diameter (ratio)')
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.savefig(output_dir / f'{file_name.stem}_plot.png')
        plt.show()

if __name__ == '__main__':
    main()

