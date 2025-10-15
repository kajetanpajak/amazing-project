import sys

import cv2 as cv
import numpy as np

from model import SegmentationModel
from utils import get_binary_mask
from datasets import get_validation_transforms

class DiameterMeasurement():
    
    def __init__(self, cap: cv.VideoCapture, model: SegmentationModel, device: str,
                transform):
        
        self.cap = cap
        self.model = model
        self.device = device
        self.transform = transform

        self.model.to(self.device)
        model.eval()

    def _find_break_frame(self, display:bool = False):
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
                                   transform=self.transform)
            
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
        mask = get_binary_mask(frame, self.model, self.device, self.transform)

        if self._break_check(mask):
            self.break_frame = high
            self.fluid_breaks = True
        else:
            ret, frame = self.cap.read()
            mask = get_binary_mask(frame, self.model, self.device, self.transform)
            if self._break_check(mask):
                self.break_frame = high + 1
                self.fluid_breaks = True
            else:
                self.fluid_breaks = False

        cv.destroyAllWindows()
            
            

    def _break_check(self, mask):
        """Checks a single frame and returns a boolean value, depending on whether
        the fluid already broke or not."""

        leftmost, rightmost = self._mask_left_right_x(mask)

        column_sums = np.sum(mask[:, leftmost:rightmost], axis=0) 
        if np.any(column_sums == 0):
            return True
        return False
    

    def _find_end_of_expansion(self, display:bool = False):
        

        low = 0
        high = self.break_frame if self.fluid_breaks else self.cap.get(cv.CAP_PROP_FRAME_COUNT)

        # get final width to compare
        self.cap.set(cv.CAP_PROP_POS_FRAMES, high-1)

        ret, frame = self.cap.read()

        if not ret:
                print("Failed to read the frame in search for end of expansion frame")
        
        mask = get_binary_mask(
            frame=frame,
            model=self.model,
            device=self.device,
            transform=self.transform)
        
        leftmost, rightmost = self._mask_left_right_x(mask)

        self.final_width = rightmost - leftmost

        # binary search for end of expansion

        while high - low > 1:
            mid = (low + high) // 2
            self.cap.set(cv.CAP_PROP_POS_FRAMES, mid-1)

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read the frame in search for end of expansion frame")
                break

            if display:
                display_frame = frame.copy()
                display_frame = cv.resize(display_frame, (512,512))
                cv.putText(display_frame, f"Frame: {mid}", (10,30), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

            mask = get_binary_mask(frame=frame,
                                   model=self.model,
                                   device=self.device,
                                   transform=self.transform)
            

            leftmost, rightmost = self._mask_left_right_x(mask)

            width = rightmost - leftmost

            if width >= self.final_width * 0.98:
                high = mid
            else:
                low = mid

            if display:
                mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                combined_display = np.hstack((display_frame, mask))

                cv.imshow('Searching for end of expansion', combined_display)
                key = cv.waitKey(200)
                if key == 27:  # ESC key to exit
                    print('Exit during searching for the end of expansion frame')
                    sys.exit()
        
        self.end_of_expansion_frame = high
        cv.destroyAllWindows()

    def _find_measurement_point(self, display:bool = False):

        if self.fluid_breaks:
            last_frame = self.break_frame - 1
        else:
            last_frame = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, max(self.end_of_expansion_frame, last_frame-10))

        narrowest_points = []

        while True:
            
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read the frame in search for the measurement point")
                break

            mask = get_binary_mask(frame=frame,
                                   model=self.model,
                                   device=self.device,
                                   transform=self.transform)
            
            leftmost, rightmost = self._mask_left_right_x(mask)

            # column sums in fluid region
            offset = 10 # offset in order to avoid finding narrowest points on edges
            leftmost, rightmost = leftmost + offset, rightmost - offset
            column_sums = mask[:, leftmost:rightmost].sum(axis=0)
            
            x_narrowest_relative = np.argmin(column_sums)
            x_narrowest = x_narrowest_relative + leftmost

            narrowest_points.append(x_narrowest)

            if display:
                frame = cv.resize(frame, (512, 512))

                measurement_col = mask[:, x_narrowest]
                diameter = np.sum(measurement_col) / 255
                top_point = np.where(measurement_col == 255)[0][0]
                cv.line(frame, (x_narrowest, top_point), (x_narrowest, top_point + int(diameter)), (0, 255, 0), 1)

                mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                combined_frames = np.hstack((frame, mask))
                cv.imshow('Searching for the measurement point', combined_frames)
                cv.waitKey(0)

            if self.cap.get(cv.CAP_PROP_POS_FRAMES) == last_frame:
                break

        self.x_measurement = int(np.mean(narrowest_points))

    def _mask_left_right_x(self, mask):
        """Returns the x-coordinates of the leftmost and rightmost white pixels of a binary mask."""
        white_coords = np.argwhere(mask==255) # list of [y, x]
        leftmost = white_coords[:, 1].min()
        rightmost = white_coords[:,1].max()

        return leftmost, rightmost



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
            classes=1
        )

    model_state_dict = torch.load('models/best_model.pth', map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)

    cap = cv.VideoCapture(os.path.join(directory, video))

    diameter_measurement = DiameterMeasurement(
        cap=cap,
        model=model,
        device=device,
        transform=get_validation_transforms((512, 512))
    )

    diameter_measurement._find_break_frame(False)
    print(diameter_measurement.break_frame)

    diameter_measurement._find_end_of_expansion(False)
    print(diameter_measurement.end_of_expansion_frame)

    diameter_measurement._find_measurement_point(True)


if __name__ == "__main__":
    main()