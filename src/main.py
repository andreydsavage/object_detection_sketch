import argparse
from pathlib import Path

from myUtils.detector_factory import DetectorFactory

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Детекция объектов с помощью разных моделей')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (cuda или cpu)')
    parser.add_argument('--source', type=str, default='data/videos/crowd.mp4',
                       help='Путь к изображению или видео')
    parser.add_argument('--model', type=str, default='frcnn', 
                       choices=['frcnn', 'frcnn_onnx', 'yolo', 'yolo_tensor', 'yolo_tensorRT'],
                       help='Тип модели для детекции (frcnn или frcnn_onnx)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Порог уверенности для детекций')
    parser.add_argument('--no-show', action='store_true',
                       help='Не показывать результат')
    parser.add_argument('--classes', type=str, default='person',
                        help='Отображать данный класс' )
    
    args = parser.parse_args()
    detector = DetectorFactory.create_detector(
        model_type=args.model,
        device=args.device,
        score_threshold=args.threshold
    )
    detector.process_video(video_path= Path(args.source),
                           show=not args.no_show,
                           classes=args.classes
                )
    
