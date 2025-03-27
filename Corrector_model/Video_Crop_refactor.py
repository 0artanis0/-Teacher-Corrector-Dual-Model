# -*- coding: utf-8 -*-
"""
定义了视频截取类VideoCrop，crop_video为视频截取方法

示例使用：从 your_video.mp4 中每 5 帧截图一次，保存到your_output_path路径下名为 "partial_view" 的文件夹中
同时截取两个部分，一个大小为 (120, 120)，位置为视频画面中心，另一个大小为 (80, 80)，位置为视频画面左上角
位置参数使用小数形式，表示在视频中心（0.5，0.5）和左上角（0，0）
同时视频会自动截取与局部图帧对应的全局图，保存在your_output_path路径下名为"global_screenshots"文件夹中

Example use: Take a screenshot every 5 frames from your_video.mp4 and save it to a folder named "partial_view" under
the your_output_path path Capture two parts at the same time, one with a size of (120, 120) and the other in the
center of the video, and the other with a size of (80, 80) in the upper left corner of the video Positional
parameters are in decimal form and are represented in the center of the video (0.5, 0.5) and in the upper left corner
(0, 0) At the same time, the video will automatically capture the global image corresponding to the local image frame
and save it in the your_output_path path named "global_screenshots" folder

    @Project : ResNet-refactor
    @File    : Video_Crop_refactor.py
    @Author  : Hongli Zhao
    @E-mail  : zhaohongli8711@outlook.com
    @Date    : 2024/01/14 17:00
    @Software: PyCharm
"""
import cv2
import os
import numpy as np


class VideoCrop:
    def __init__(self, video_path, output_path, rect_sizes, rect_positions, interval=5):
        """
        Initializes the VideoCrop object.

        Args:
        - video_path (str): Path to the input video file.
        - output_path (str): Path to the main output folder.
        - rect_sizes (list): List of tuples representing the sizes of rectangular regions to capture.
        - rect_positions (list): List of tuples representing the positions of rectangular regions to capture.
        - interval (int, optional): Number of frames to skip before capturing a screenshot. Default is 5.
        """
        self.video_path = video_path
        self.partial_view_patch = os.path.join(output_path, "partial_view")
        self.global_screenshots_path = os.path.join(output_path, "global_screenshots")
        self.rect_sizes = rect_sizes
        self.rect_positions = rect_positions
        self.interval = interval

        # Create folders to save screenshots
        os.makedirs(self.partial_view_patch, exist_ok=True)
        os.makedirs(self.global_screenshots_path, exist_ok=True)

    def crop_video(self):
        """
        Opens the video file, captures frames, processes them, and saves screenshots to specified folders.
        """

        # Open the video file
        cap = cv2.VideoCapture(self.video_path)

        # Get the video's frame rate and frame size
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a window to display cropped frames
        cv2.namedWindow('Cropped Frames', cv2.WINDOW_NORMAL)

        # Process the video frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Video has ended

            # Capture a screenshot every interval frames
            if frame_count % self.interval == 0:
                # Create a blank canvas for merging screenshots
                merged_frame = np.zeros((height, width, 3), dtype=np.uint8)

                for rect_size, rect_position in zip(self.rect_sizes, self.rect_positions):
                    # Calculate actual coordinates
                    x = int((width - rect_size[0]) * rect_position[0])
                    y = int((height - rect_size[1]) * rect_position[1])

                    # Capture the specified region
                    cropped_frame = frame[y:y + rect_size[1], x:x + rect_size[0]]

                    # Save the screenshot to the specified folder
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    output_path = os.path.join(self.partial_view_patch,
                                               f'frame_{frame_number}_position_{x}_{y}_RectSize_{rect_size[0]}_{rect_size[1]}.png')
                    cv2.imwrite(output_path, cropped_frame)

                    # Merge the screenshot onto the canvas
                    merged_frame[y:y + rect_size[1], x:x + rect_size[0]] = cropped_frame

                    # Display the cropped frame
                    cv2.imshow('Cropped Frames', merged_frame)

                global_frame = frame
                global_output_path = os.path.join(self.global_screenshots_path,
                                                  f'frame_{frame_number}.png')
                cv2.imwrite(global_output_path, global_frame)

            frame_count += 1

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Example use:
if __name__ == '__main__':
    rect_sizes = [(120, 120), (80, 80)]
    rect_positions = [(0.5, 0.5), (0, 0)]
    video_path = 'your_video.mp4'
    output_path = 'your_output_path'
    interval = 5

    video = VideoCrop(video_path, output_path, rect_sizes, rect_positions, interval)    # 类实例化
    video.crop_video()      # 调用crop_video方法
