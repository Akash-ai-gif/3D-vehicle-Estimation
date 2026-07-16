import os

print("=== Processing Top Angle Video (traffic_video.mp4) ===")
os.system("python src/main.py --input traffic_video.mp4 --output traffic_video_output.avi")
os.system(".\\ffmpeg.exe -y -i traffic_video_output.avi -c:v libx264 -preset fast -crf 22 traffic_video_output_h264.mp4")

print("\n=== Processing Dense Traffic Video (raw_video.mp4) ===")
os.system("python src/main.py --input raw_video.mp4 --output result_3d_boxes.avi")
os.system(".\\ffmpeg.exe -y -i result_3d_boxes.avi -c:v libx264 -preset fast -crf 22 traffic_result_3d_h264.mp4")

print("\n=== Processing KITTI Dataset (1106 frames) ===")
os.system("python src/main.py --input kitti_dataset\\2011_09_30\\2011_09_30_drive_0027_sync\\image_02\\data --output kitti_result_3d.avi --fps 10.0")
os.system(".\\ffmpeg.exe -y -i kitti_result_3d.avi -c:v libx264 -preset fast -crf 22 kitti_result_3d_h264.mp4")

print("\nAll videos processed and encoded successfully!")
