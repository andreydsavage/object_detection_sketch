import torch
import torchvision
import cv2
import json
import os
import time
from collections import defaultdict
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')

class Detector:
    """Базовый класс детектора объектов"""
    def __init__(self, model_type='frcnn', device=None, score_threshold=0.5):
        """
        Инициализация детектора
        
        Args:
            model_type: тип модели ('frcnn' или 'yolo')
            device: устройство для вычислений ('cuda' или 'cpu')
            score_threshold: порог уверенности для детекций
        """
        self.model_type = model_type.lower()
        self.score_threshold = score_threshold
        self.output_path = Path('data/output/')
        os.makedirs(self.output_path, exist_ok=True)
        
        # Определяем устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Используется устройство: {self.device}")
        
        # Инициализируем выбранную модель
        if self.model_type == 'frcnn':
            self._init_frcnn()
        elif self.model_type == 'yolo':
            self._init_yolo()
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Поддерживаемые: 'frcnn', 'yolo'")
    
    def _init_frcnn(self):
        """Инициализация Faster R-CNN модели"""
        print("Загрузка Faster R-CNN модели...")
        weights_path = Path('src/models/fasterrcnn_resnet50_fpn.pth')
        if weights_path.exists():
            print('Используем локальную модель')
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,)
            state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict)

        else:
            print('Загружаем модель из интернета')
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO классы для Faster R-CNN
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def _init_yolo(self):
        """Инициализация YOLO модели"""
        print("Загрузка YOLO модели...")
        model_path = Path('src/models/yolo26n.pt')
        if model_path.exists():
            print('Используем локальную модель')
            self.model = YOLO(model_path)
        else:
            print('Загружаем модель из интернета')
            self.model = YOLO('yolo26n.pt')
        self.model.to(self.device)
    
    def preprocess(self, frame):
        """
        Предобработка изображения
        
        Args:
            frame: изображение в формате PIL.Image, numpy.ndarray или путь к файлу
            
        Returns:
            torch.Tensor: тензор изображения
        """
        # Если передан путь к файлу
        if isinstance(frame, (str, Path)):
            frame = Image.open(frame)
        
        # Если передан numpy array (например, из OpenCV)
        if isinstance(frame, np.ndarray):
            # Конвертируем BGR в RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        
        # Возвращаем PIL Image для дальнейшей обработки
        return frame
    
    def predict(self, frame):
        """
        Выполнение детекции объектов
        
        Args:
            frame: входное изображение
            
        Returns:
            tuple: (boxes, scores, labels, class_names)
        """
        # Предобработка изображения
        pil_image = self.preprocess(frame)
        
        if self.model_type == 'frcnn':
            return self._predict_frcnn(pil_image)
        elif self.model_type == 'yolo':
            return self._predict_yolo(pil_image)
    
    def _predict_frcnn(self, pil_image):
        """Предсказание с использованием Faster R-CNN"""
        # Преобразуем в тензор
        image_tensor = torchvision.transforms.functional.to_tensor(pil_image)
        image_tensor = image_tensor.to(self.device)
        
        # Выполняем предсказание
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Извлекаем результаты
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Фильтруем по порогу уверенности
        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Получаем имена классов
        
        class_names = [self.class_names[label] for label in labels]
        
        return boxes, scores, labels, class_names
    
    def _predict_yolo(self, pil_image):
        """Предсказание с использованием YOLO"""

        results = self.model(pil_image, imgsz =1920)# Можно настроить размер
        
        # Извлекаем данные
        boxes = results[0].boxes.xyxy
        boxes = boxes.cpu().numpy()
        scores = np.array([box.conf[0].item() for box in results[0].boxes])
        labels = np.array([int(box.cls) for box in results[0].boxes])
        
                # Фильтруем по порогу уверенности
        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Получаем имена классов
        class_names = [results[0].names[label] for label in labels]
        
        
        return boxes, scores, labels, class_names
    
    def visualize(self, frame, boxes, scores, labels, class_names,
                  vis_class, 
                  thickness=2, font_scale=0.5, color=(0, 255, 0)):
        """
        Визуализация результатов детекции
        
        Args:
            frame: исходное изображение
            boxes: координаты bounding box
            scores: уверенность детекций
            labels: метки классов
            class_names: имена классов
            thickness: толщина линий bounding box
            font_scale: масштаб шрифта
            color: цвет bounding box (BGR)
            
        Returns:
            numpy.ndarray: изображение с визуализацией детекций
        """
        
        # Рисуем bounding boxes
        for box, score, label, class_name in zip(boxes, scores, labels, class_names):
            if class_name in vis_class:
                x1, y1, x2, y2 = box.astype(int)
                
                # Рисуем прямоугольник
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Добавляем текст с классом и уверенностью
                text = f"{class_name}: {score:.2f}"
                
                # Получаем размер текста
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Рисуем фон для текста
                cv2.rectangle(frame, 
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Добавляем текст
                cv2.putText(frame, text, (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return frame
    
    def process_video(self, video_path, show=True, fps=30, classes = ['person']):
        """
        Обработка видеофайла
        
        Args:
            video_path: путь к входному видео
            output_path: путь для сохранения результата (если None - не сохранять)
            show: показывать ли результат в реальном времени
            fps: FPS для выходного видео
        """
        cap = cv2.VideoCapture(video_path if isinstance(video_path, str) else str(video_path))
        
        if not cap.isOpened():
            print(f"Ошибка открытия видео: {video_path}")
            return
        
        # Получаем параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Настраиваем видеозапись если нужно
        writer = None
        output_path = self.output_path
        output_video_path = output_path / f'{self.model_type}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path.with_suffix('.mp4'), fourcc, fps, (new_width, new_height))
        
        frame_count = 0
        total_detections = defaultdict(int)
        latencies = []
        print(f"Начата обработка видео: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.perf_counter()
            # Выполняем детекцию
            boxes, scores, labels, class_names = self.predict(frame)
            for label in class_names:
                total_detections[label] += 1 
            
            
            # Визуализируем результаты
            result_frame = self.visualize(frame, boxes, scores, labels, class_names,
                                          classes)
            result_frame = cv2.resize(result_frame, (new_width, new_height))
            # Сохраняем если нужно
            if writer:
                writer.write(result_frame)
            
            # Показываем если нужно
            if show:
                cv2.imshow(f'Detector - {self.model_type.upper()}', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}")

            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # в миллисекундах
            latencies.append(latency)

        json_path = Path(output_video_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(dict(total_detections), f)

        stats = {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'fps': 1000 / np.mean(latencies),  # FPS
            'num_runs': len(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        }
        stats_path = (Path(str(output_video_path) + f'_{self.device}_stats')).with_suffix('.json')
        with open(stats_path, 'w') as f:
            json.dump(dict(stats), f)

        # Освобождаем ресурсы
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"Обработка завершена. Всего кадров: {frame_count}")



# Пример использования
if __name__ == "__main__":
    import argparse
    
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Детекция объектов с помощью разных моделей')
    parser.add_argument('--model', type=str, default='frcnn', 
                       choices=['frcnn', 'yolo'],
                       help='Тип модели для детекции (frcnn или yolo)')
    parser.add_argument('--source', type=str, default='data/videos/crowd.mp4',
                       help='Путь к изображению или видео')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Порог уверенности для детекций')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (cuda или cpu)')
    parser.add_argument('--no-show', action='store_true',
                       help='Не показывать результат')
    parser.add_argument('--classes', type=str, default='person',
                        help='Отображать данный класс' )
    
    args = parser.parse_args()
    
    # Инициализация детектора
    print(f"Инициализация {args.model.upper()} модели...")
    detector = Detector(
        model_type=args.model,
        device=args.device,
        score_threshold=args.threshold
    )
    
    # Проверяем тип входного файла
    source_path = Path(args.source)
            
    if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Обработка видео
        print(f"Обработка видео: {args.source}")
        detector.process_video(
            video_path=args.source,
            show=not args.no_show,
            classes=args.classes
        )
    else:
        print(f"Неподдерживаемый формат файла: {source_path.suffix}")