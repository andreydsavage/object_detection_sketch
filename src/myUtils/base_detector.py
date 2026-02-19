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
        self.score_threshold = score_threshold
        self.output_path = Path('data/output/')
        os.makedirs(self.output_path, exist_ok=True)
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Используется устройство: {self.device}")
        
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
        keep = scores >= self.score_threshold
        return boxes[keep], scores[keep], labels[keep], [self.class_names[label] for label in labels[keep]]

    @staticmethod
    def _show_fps(img, fps):
        text = f"FPS: {fps:.1f}"
        color = (0, 255, 0) if fps >= 60 else (0, 255, 255) if fps >= 30 else (0, 0, 255)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img

    def visualize(self, frame, boxes, scores, labels, class_names, vis_class, 
                  thickness=2, font_scale=0.5, color=(0, 255, 0)):
        """Оптимизированная визуализация"""
        for box, score, class_name in zip(boxes, scores, class_names):
            if class_name in vis_class:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                text = f"{class_name}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - baseline - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        return frame
    
    def process_video(self, video_path, show=True, target_fps=30, classes=['person'], scale=1):
        """Оптимизированная обработка видео"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Ошибка открытия видео: {video_path}")
            return
        video_filename = video_path.stem
        
        # Параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width, new_height = int(width * scale), int(height * scale)
        
        # Настройка записи
        output_video_path = self.output_path / f'{self.model_type}_{video_filename}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video_path), fourcc, target_fps, (new_width, new_height))
        
        frame_count = 0
        total_detections = defaultdict(int)
        latencies = []
        
        # Для FPS
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print(f"Начата обработка видео: {video_path}")
        
        # Оптимизация: предварительное создание массивов для результатов
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.perf_counter()
            
            # Детекция
            boxes, scores, labels, class_names = self.predict(frame)
            
            # Подсчет детекций (оптимизировано)
            for name in class_names:
                total_detections[name] += 1
            
            inference_time = (time.perf_counter() - start_time) * 1000
            latencies.append(inference_time)
            
            # FPS расчет
            fps_counter += 1
            if fps_counter >= 30:
                current_fps = 1000 / ((time.perf_counter() - fps_start_time) * 1000) * 30
                fps_counter = 0
                fps_start_time = time.perf_counter()
            
            # Визуализация
            if show or writer:
                result_frame = self.visualize(frame, boxes, scores, labels, class_names, classes)
                result_frame = self._show_fps(result_frame, current_fps )
                
                if new_width != width or new_height != height:
                    result_frame = cv2.resize(result_frame, (new_width, new_height))
                
                if writer:
                    writer.write(result_frame)
                
                if show:
                    cv2.imshow(f'Detector - {self.model_type.upper()}', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            if frame_count % 100 == 0:  # Увеличил интервал вывода
                print(f"Обработано кадров: {frame_count}, средний FPS для inference: {1000/np.mean(latencies[-100:]):.1f}")
        
        # Сохранение статистики
        self._save_statistics(total_detections, latencies, output_video_path)
        
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"Обработка завершена. Всего кадров: {frame_count}")
    
    def _save_statistics(self, total_detections, latencies, output_video_path):
        """Сохранение статистики"""
        json_path = output_video_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(dict(total_detections), f)
        
        stats = {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'fps_model_inference': 1000 / np.mean(latencies),
            'num_runs': len(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        }
        
        stats_path = output_video_path.with_name(f"{output_video_path.stem}_{self.device}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f)