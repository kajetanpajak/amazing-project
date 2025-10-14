import sys

import cv2 as cv
import numpy as np

from model import SegmentationModel
from utils import get_binary_mask

class DiameterMeasurement():
    
    def __init__(self, cap: cv.VideoCapture, model: SegmentationModel, device: str):
        
        self.cap = cap
        self.model = model
        self.device = device
        self.validation_transform = model.validation_transform

        self.model.to(self.device)
        model.eval()

    def _get_break_frame(self, display:bool = False):
        """
        Checks if the fluid breaks in the video. If it does, finds the frame index where it happens.
        """

        low, high = 0, int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

        while high - low > 1:
            
            mid = (low + high) // 2
            self.cap.set(cv.CAP_PROP_POS_FRAMES, mid-1)

            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read the frame in search for break frame")
                break

            if display:
                display_frame = frame.copy()
                display_frame = cv.resize(display_frame, (512,512))
                cv.putText(display_frame, f"Frame: {mid}", (10,30), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

            mask = get_binary_mask(frame=frame,
                                   model=self.model,
                                   device=self.device,
                                   transform=self.model.validation_transform)
            
            if self._break_check(mask):
                high = mid
            else: 
                low = mid

            if display:
                mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                combined = np.hstack((display_frame, mask))

                cv.imshow('Searching for break frame', combined)
                key_code = cv.waitKey(200)
                if key_code == ord('q'):
                    print('Exit during search for break frame')
                    sys.exit()
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, high-1)
        ret, frame = self.cap.read()
        mask = get_binary_mask(frame, self.model, self.device, self.model.validation_transform)

        if self._break_check(mask):
            self.break_frame = high
            self.fluid_breaks = True
        else:
            ret, frame = self.cap.read()
            mask = get_binary_mask(frame, self.model, self.device, self.model.validation_transform)
            if self._break_check(mask):
                self.break_frame = high + 1
                self.fluid_breaks = True
            else:
                self.fluid_breaks = False
            
            

    def _break_check(self, mask):
        """Checks a single frame and returns a boolean value, depending on whether
        the fluid already broke or not."""

        white_coords = np.argwhere(mask==255) # list of [y, x]
        left_most = white_coords[:, 1].min()
        right_most = white_coords[:,1].max()

        column_sums = np.sum(mask[:, left_most:right_most], axis=0) 
        if np.any(column_sums == 0):
            return True
        return False
    

    def _get_end_of_expansion(self, display:bool = False):
        

        low = 0
        high = self.break_frame if self.fluid_breaks else self.cap.get(cv.CAP_PROP_FRAME_COUNT)

        # get final width to compare
        self.cap.set(cv.CAP_PROP_POS_FRAMES, high-1)

        ret, frame = self.cap.read()
        mask = get_binary_mask(
            frame=frame,
            model=self.model,
            device=self.device,
            transform=self.model.validation_transform)
        
        white_coords = np.argwhere(mask==255) # list of [y, x]
        left_most = white_coords[:, 1].min()
        right_most = white_coords[:,1].max()

        self.final_width = right_most - left_most

        # binary search for end of expansion

        while high - low > 1:
            mid = (low + high) / 2
            self.cap.set(cv.CAP_PROP_POS_FRAMES, mid-1)

            ret, frame = self.cap.read()

            if display:
                display_frame = frame.copy()
                display_frame = cv.resize(display_frame, (512,512))
                cv.putText(display_frame, f"Frame: {mid}", (10,30), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

            mask = get_binary_mask(frame=frame,
                                   model=self.model,
                                   device=self.device,
                                   transform=self.model.validation_transform)
            

            white_coords = np.argwhere(mask==255) # list of [y, x]
            left_most = white_coords[:, 1].min()
            right_most = white_coords[:,1].max()

            width = right_most - left_most

            if width >= self.final_width * 0.98:
                high = mid
            else:
                low = mid

            if display:
                mid_frame_mask = cv.cvtColor(mid_frame_mask, cv.COLOR_GRAY2BGR)
                combined_display = np.hstack((display_frame, mid_frame_mask))

                cv.imshow('Searching for end of expansion', combined_display)
                key = cv.waitKey(200)
                if key == 27:  # ESC key to exit
                    print('Exit during searching for the end of expansion frame')
                    sys.exit()

            

        


def main():
    import torch
    import os
    from pathlib import Path

    directory = Path('~/Videos/fluid_videos/pendrive2').expanduser()
    video = Path('próbki 13.2/13.2 odległość 4 mm prędkość 10 1.MP4')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegmentationModel(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            image_size=(512,512)
        )

    model_state_dict = torch.load('models/best_model.pth', map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)

    cap = cv.VideoCapture(os.path.join(directory, video))

    diameter_measurement = DiameterMeasurement(
        cap=cap,
        model=model,
        device=device
    )

    diameter_measurement._get_break_frame(True)
    print(diameter_measurement.break_frame)

    cap.set(cv.CAP_PROP_POS_FRAMES, diameter_measurement.break_frame-1)
    ret, frame = cap.read()
    frame = cv.resize(frame, (512, 512))

    cv.imshow('xd', frame)
    cv.waitKey(0)


if __name__ == "__main__":
    main()