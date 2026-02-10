import onnxruntime
import cv2
import time
import torchvision

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import json
from pathlib import Path
import os
import torch



class BaseDetector(ABC):

    def __init__(self, device=None, score_threshold=0.5):
        """
        Инициализация детектора
        
        Args:
            device: устройство для вычислений ('cuda' или 'cpu')
            score_threshold: порог уверенности для детекций
        """
        self.score_threshold = score_threshold
        self.output_path = Path('data/output/')
        os.makedirs(self.output_path, exist_ok=True)
        
        # Определяем устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Используется устройство: {self.device}")
        
        # Инициализация модели
        self.model = None
        self.class_names = None
        self._init_model()


    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    def _filter_results(self, boxes, scores, labels):
        """
        Фильтрация результатов по порогу уверенности
        
        Args:
            boxes: координаты bounding boxes
            scores: уверенности детекций
            labels: метки классов
            class_names: имена классов
            
        Returns:
            tuple: отфильтрованные результаты
        """
        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        class_names = [self.class_names[label] for label in labels]
        return boxes, scores, labels, class_names

    def _show_fps(self, img, fps):

        text = f"FPS: {fps:.1f}"
        position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        if fps >= 60:
            color = (0, 255, 0)  # Зеленый
        elif fps >= 30:
            color = (0, 255, 255)  # Желтый
        else:
            color = (0, 0, 255)  # Красный
        # Основной текст
        cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        return img

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
    
    def process_video(self, video_path, show=True, fps=30, classes = ['person'], scale = 1):
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
            end_time = time.perf_counter()
            
            # Визуализируем результаты
            result_frame = self.visualize(frame, boxes, scores, labels, class_names,
                                          classes)

            # inference_time = (time.perf_counter() - start_time) * 1000  # мс
            fps = 1000 / np.mean(latencies[-1:])                             
            result_frame = self._show_fps(result_frame, fps)                              

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



