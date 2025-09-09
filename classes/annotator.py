import os
import sys
import csv
import cv2
import numpy as np
import pandas as pd

class Annotator:
    def __init__(self, video_path, num_elements, frame_skip, output_folder, annotation_window_width = 1000):
        self.output_folder = output_folder
        self.video_path = video_path
        self.num_elements = num_elements
        self.frame_skip = frame_skip
        self.rects = []
        self.skips = []
        self.skips_total = []
        self.current_rect = []
        self.drawing = False
        self.frame = None
        self.clone = None
        self.annotations = []  # List of (frame_idx, rects)
        self.frame_indices = []
        self.annotation_window_width = annotation_window_width

        self.colors = [(0, 255, 0), 
                       (255, 0, 0), 
                       (0, 0, 255), 
                       (255, 255, 0), 
                       (0, 255, 255),
                       (255, 0, 255), 
                       (255, 128, 0), 
                       (128, 255, 0), 
                       (0, 128, 255), 
                       (128, 0, 255)] * 10

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = False
            self.rects.append([(-1, -1), (-1, -1)])
            self.skips.append(True)
            self.current_rect = []
            self.clone = self.frame.copy()
            for idx, rect in enumerate(self.rects):
                cv2.rectangle(self.clone, rect[0], rect[1], self.colors[idx], 2)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(self.rects) < self.num_elements:
                self.drawing = True
                self.current_rect = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.clone = self.frame.copy()
                cv2.rectangle(self.clone, self.current_rect[0], (x, y), self.colors[len(self.rects)], 2)
                for idx, rect in enumerate(self.rects):
                    cv2.rectangle(self.clone, rect[0], rect[1], self.colors[idx], 2)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.current_rect.append((x, y))
                self.skips.append(False)
                self.rects.append(tuple(self.current_rect))
                self.current_rect = []
                self.clone = self.frame.copy()
                for idx, rect in enumerate(self.rects):
                    cv2.rectangle(self.clone, rect[0], rect[1], self.colors[idx], 2)

    def annotate_frame(self, frame, frame_idx):
        self.frame = frame.copy()
        self.clone = frame.copy()
        self.rects = []
        self.skips = []
        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", self.draw_rectangle)
        self.skips_total.append((self.skips, frame_idx))
        while True:
            cv2.imshow("Annotate", self.clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.rects = []
                self.clone = self.frame.copy()
            elif key == ord('q'):
                sys.exit(0)
            elif len(self.rects) == self.num_elements:  # Enter key
                break
        cv2.destroyWindow("Annotate")
        self.annotations.append([list(rect) for rect in self.rects])
        self.frame_indices.append(frame_idx)

    def interpolate_rects(self, rects1, rects2, steps):
        interpolated = []
        for i in range(self.num_elements):
            r1 = np.array(rects1[i])
            r2 = np.array(rects2[i])
            step_rects = []
            for s in range(1, steps):
                interp = r1 + (r2 - r1) * (s / steps)
                interp = interp.astype(int)
                step_rects.append([(int(interp[0][0]), int(interp[0][1])), (int(interp[1][0]), int(interp[1][1]))])
            interpolated.append(step_rects)
        # Transpose to get list of rects per frame
        return [ [interpolated[j][i] for j in range(self.num_elements)] for i in range(steps-1) ]

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return
        original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for annotation window
            height, width = frame.shape[:2]
            scale = self.annotation_window_width / width
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (self.annotation_window_width, new_height))

            self.annotate_frame(resized_frame, frame_idx)
            frame_idx += self.frame_skip

        # Resize annotations to original frame size
        for i in range(len(self.annotations)):
            for j in range(len(self.annotations[i])):
                rect = self.annotations[i][j]
                rect[0] = (int(rect[0][0] / scale), int(rect[0][1] / scale))
                rect[1] = (int(rect[1][0] / scale), int(rect[1][1] / scale))

        # Interpolate rectangles for skipped frames
        all_rects = []
        for i in range(len(self.annotations)-1):
            start_idx = self.frame_indices[i]
            end_idx = self.frame_indices[i+1]
            steps = end_idx - start_idx
            all_rects.append(self.annotations[i])
            interpolated = self.interpolate_rects(self.annotations[i], self.annotations[i+1], steps)
            all_rects.extend(interpolated)
        all_rects.append(self.annotations[-1])

        # Draw rectangles on all frames and save video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(os.path.join(self.output_folder, 'annotated_output.mp4'), fourcc, fps, (width, height))

        # Storing annotated file
        for _, rects in enumerate(all_rects):
            ret, frame = cap.read()
            if not ret:
                break
            for idx, rect in enumerate(rects):
                cv2.rectangle(frame, rect[0], rect[1], self.colors[idx], 2)
            out.write(frame)
        cap.release()
        out.release()

        # Dealing with skipped frames
        currently_skipped = [True for _ in range(self.num_elements)]
        skipped_by_frame = [[] for _ in range(self.num_elements)]
        current_annotation = 0
        for f in range(total_frames):
            if current_annotation >= len(self.skips_total):
                break
            if f == self.skips_total[current_annotation][1]:
                for i in range(self.num_elements):
                    if current_annotation < len(self.skips_total)-1:
                        if self.skips_total[current_annotation + 1][0][i] == True:
                            currently_skipped[i] = True
                        else:
                            currently_skipped[i] = self.skips_total[current_annotation][0][i]
                current_annotation += 1

            for i in range(self.num_elements):
                skipped_by_frame[i].append(currently_skipped[i])

        # Save annotations to CSV
        with open(os.path.join(self.output_folder, 'annotations.csv'), 'w', newline='') as csvfile:
            fieldnames = ['frame'] + [f'element_{i}' for i in range(self.num_elements)] + [f'skipped_element_{i}' for i in range(self.num_elements)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, rects in enumerate(all_rects):
                row = {'frame': idx}
                for i, rect in enumerate(rects):
                    row[f'element_{i}'] = (rect[0][0], rect[0][1], rect[1][0], rect[1][1])
                    row[f'skipped_element_{i}'] = skipped_by_frame[i][idx]
                writer.writerow(row)
        
        # Create location CSV
        position_data = pd.read_csv(os.path.join(self.output_folder, 'annotations.csv'))

        elements = []
        sides = []
        for e in range(self.num_elements):
            # Removing skipped frames
            pos_copy = position_data.copy()
            pos_copy = pos_copy[[f'element_{e}', f'skipped_element_{e}']]
            pos_copy = pos_copy[~pos_copy[f'skipped_element_{e}']]

            # Sepparating coordinates
            pos_copy[['x1', 'y1', 'x2', 'y2']] = pos_copy[f'element_{e}'].str.extract(r'\((\d+), (\d+), (\d+), (\d+)\)')
            pos_copy[['x1', 'y1', 'x2', 'y2']] = pos_copy[['x1', 'y1', 'x2', 'y2']].astype(int)
            pos_copy['averageX'] = (pos_copy['x1'] + pos_copy['x2']) / 2

            # Calculating sides
            pos_copy['side'] = pos_copy['averageX'].apply(lambda x: 'left' if x < original_frame_width / 2 else 'right')
            elements.append(f'element_{e}')

            unique_sides = pos_copy['side'].unique()
            if len(unique_sides) > 1:
                sides.append('both')
            else:
                sides.append(unique_sides[0])

        pd.DataFrame({'element': elements, 'side': sides}).to_csv(os.path.join(self.output_folder, 'sides.csv'), index=False)