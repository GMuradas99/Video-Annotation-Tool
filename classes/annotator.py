import os
import sys
import csv
import cv2
import numpy as np

class Annotator:
    def __init__(self, video_path, num_elements, frame_skip, output_folder, annotation_window_width = 1000):
        self.output_folder = output_folder
        self.video_path = video_path
        self.num_elements = num_elements
        self.frame_skip = frame_skip
        self.rects = []
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
                self.rects.append(tuple(self.current_rect))
                self.current_rect = []
                self.clone = self.frame.copy()
                for idx, rect in enumerate(self.rects):
                    cv2.rectangle(self.clone, rect[0], rect[1], self.colors[idx], 2)

    def annotate_frame(self, frame, frame_idx):
        self.frame = frame.copy()
        self.clone = frame.copy()
        self.rects = []
        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", self.draw_rectangle)
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
                step_rects.append([tuple(interp[0]), tuple(interp[1])])
            interpolated.append(step_rects)
        # Transpose to get list of rects per frame
        return [ [interpolated[j][i] for j in range(self.num_elements)] for i in range(steps-1) ]

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        annotated_frames = []

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

        # Save annotations to CSV
        with open(os.path.join(self.output_folder, 'annotations.csv'), 'w', newline='') as csvfile:
            fieldnames = ['frame'] + [f'element_{i}' for i in range(self.num_elements)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx, rects in enumerate(all_rects):
                row = {'frame': idx}
                for i, rect in enumerate(rects):
                    row[f'element_{i}'] = f'{rect[0][0]},{rect[0][1]},{rect[1][0]},{rect[1][1]}'
                writer.writerow(row)