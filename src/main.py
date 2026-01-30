from utils.inference import process_video
from ultralytics import YOLO

VIDEO = '/home/andrey/repositories/object_detection_sketch/data/videos/crowd.mp4'
MODEL = YOLO('models/yolo26n.pt')
TARGET_CLASSES = [0] #person 0 , umbrella 25
OUTPUT_PATH = '/home/andrey/repositories/object_detection_sketch/data/output/crowd.mp4'

if __name__ == "__main__":
    process_video(VIDEO, model=MODEL, target_classes=TARGET_CLASSES, show=False)
    print('Video is processed!')
