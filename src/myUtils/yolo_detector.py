
import torch
import torchvision
# import onnxruntime
import cv2


from myUtils.base_detector import BaseDetector

class Yolo_v5s_Detector(BaseDetector):  
    def _init_model(self):
        self.model_type = 'yolo_v5s'
        self.model = torch.hub.load("ultralytics/yolov5", 'yolov5s')
        # print('Загружена yolo')
        self.model.to(self.device)
        self.model.eval()
        print('Загружена yolo')

        self.class_names = list(self.model.names.values())

    def predict(self, frame):
        # Преобразуем в нужный формат
        image_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image_tensor = image_tensor.to(self.device)

        # Выполняем предсказание
        with torch.no_grad():
            predictions = self.model([image_tensor])
            predictions = predictions.xyxy
        
        # Извлекаем результаты
        boxes = predictions[0][:,:4].cpu().numpy()
        scores = predictions[0][:,4].cpu().numpy()
        labels = predictions[0][:,5].cpu().numpy().astype(int)

        boxes, scores, labels, class_names = self._filter_results(boxes, scores, labels)
        return boxes, scores, labels, class_names

from myUtils.base_detector import BaseDetector

import cv2
import numpy as np
import torch
from torchvision.ops import nms

class Yolo_v5s_tensor_Detector(BaseDetector):  
    def _init_model(self):
        self.model_type = 'yolo_tensor'
        self.model = torch.hub.load("ultralytics/yolov5", 'yolov5s')
        # print('Загружена yolo')
        self.model.to(self.device)
        self.model.eval()
        print('Загружена yolo')

        self.class_names = list(self.model.names.values())

    def predict(self, frame):
        # Преобразуем в нужный формат
        image_tensor, pad_info = self._prepare_frame_for_yolo(frame, target_size = 640)
        # image_tensor = image_tensor.to(self.device)

        # Выполняем предсказание
        with torch.no_grad():
            predictions = self.model(image_tensor)
            # predictions = predictions.xyxy
        
        # Извлекаем результаты
        boxes = predictions[0][:,:4].cpu() # координаты [1, 25200, 4]
        objectness = predictions[0][:,4].cpu()
        class_probs = predictions[0][:,5:].cpu()

        boxes = self._raw_to_xyxy(boxes)
        max_class_probs, max_class_ids = torch.max(class_probs, dim=-1)
        total_confidence =  max_class_probs * objectness

        confidence_threshold = self.score_threshold
        valid_detections = total_confidence > confidence_threshold
        boxes = boxes[valid_detections]
        labels = max_class_ids[valid_detections]
        scores = total_confidence[valid_detections]

        # Применяем NMS
        keep_indices = nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=0.5  # порог IoU (обычно 0.4-0.5)
        )
        boxes = boxes[keep_indices].numpy()
        labels = labels[keep_indices].numpy()
        scores = scores[keep_indices].numpy()

        boxes, scores, labels, class_names = self._filter_results(boxes, scores, labels)
        boxes = self._correct_bbox_coordinates(boxes,pad_info)
        return boxes, scores, labels, class_names
    
    def _prepare_frame_for_yolo(self, frame, target_size=640):
        """
        Правильная предобработка кадра из cv2.VideoCapture для YOLO
        
        Args:
            frame: кадр из cv2.VideoCapture (numpy array в формате BGR)
            target_size: целевой размер для YOLO
        
        Returns:
            tensor: подготовленный тензор для модели [1, 3, target_size, target_size]
            pad_info: информация о паддинге для коррекции координат
        """
 
        
        # Конвертация BGR -> RGB (так как OpenCV читает в BGR, а YOLO ожидает RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = frame_rgb.shape[:2]
        
        # 1. Letterbox (сохранение пропорций с паддингом)
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Ресайз с сохранением пропорций
        img_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Создаем изображение с паддингом
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # серый цвет (114,114,114) как в YOLO
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        img_padded[paste_y:paste_y + new_h, paste_x:paste_x + new_w, :] = img_resized
        
        # 2. Конвертация в тензор (как в YOLO)
        img_array = img_padded.transpose(2, 0, 1)  # HWC -> CHW
        img_array = img_array.astype(np.float32) / 255.0  # нормализация [0, 1]
        tensor = torch.from_numpy(img_array).unsqueeze(0)  # добавить batch
        
        # Сохраняем информацию о паддинге для коррекции координат
        pad_info = {
            'scale': scale,
            'pad_x': paste_x,
            'pad_y': paste_y,
            'orig_size': (orig_w, orig_h)
        }
        
        return tensor, pad_info

    def _raw_to_xyxy(self,raw_boxes, img_width=640, img_height=640):
        """
        raw_boxes: тензор [N, 85] в формате [cx, cy, w, h, obj, class_probs...]
        """
        # Извлекаем координаты (нормализованные)
        cx, cy, w, h = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
        
        # Конвертируем в абсолютные пиксели
        cx_abs = cx 
        cy_abs = cy 
        w_abs = w 
        h_abs = h
        
        # Преобразуем в [x1, y1, x2, y2]
        x1 = cx_abs - w_abs/2
        y1 = cy_abs - h_abs/2
        x2 = cx_abs + w_abs/2
        y2 = cy_abs + h_abs/2
        
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _correct_bbox_coordinates(self, bboxes, pad_info):
        """
        Корректировка координат bbox после удаления паддинга
        
        Args:
            bboxes: массив bounding boxes в формате [x1, y1, x2, y2] в координатах паддинга
            pad_info: информация о паддинге из prepare_frame_for_yolo
        
        Returns:
            bboxes в исходных координатах кадра
        """
        if len(bboxes) == 0:
            return bboxes
        
        scale = pad_info['scale']
        pad_x = pad_info['pad_x']
        pad_y = pad_info['pad_y']
        
        # Убираем паддинг
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_x) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_y) / scale
        
        # Клиппинг к границам исходного изображения
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, pad_info['orig_size'][0])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, pad_info['orig_size'][1])
        
        return bboxes
    


import torch_tensorrt
class Yolo_v5s_tensorRT_Detector(BaseDetector):
    _preloaded_model = None
    _preloaded_class_names = None
    
    @classmethod
    def preload_model(cls):
        """Предварительная загрузка модели в главном потоке"""
        if cls._preloaded_model is None:
            import torch_tensorrt
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            base_model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained=True, verbose=False)
            cls._preloaded_class_names = list(base_model.names.values())
            cls._preloaded_model = torch.export.load("/home/andrey/repositories/object_detection_sketch/src/models/trt.pt2").module()
            print('Загружена yolo с TensorRT (preload)')
        return cls._preloaded_model, cls._preloaded_class_names
    
    @classmethod
    def clear_preloaded(cls):
        """Очистка предзагруженной модели"""
        cls._preloaded_model = None
        cls._preloaded_class_names = None
    
    def __init__(self, device=None, score_threshold=0.5, preloaded_model=None):
        if preloaded_model is not None:
            Yolo_v5s_tensorRT_Detector._preloaded_model = preloaded_model[0]
            Yolo_v5s_tensorRT_Detector._preloaded_class_names = preloaded_model[1]
        super().__init__(device=device, score_threshold=score_threshold)
    
    def _init_model(self):
        self.model_type = 'yolo_tensorRT'
        
        if Yolo_v5s_tensorRT_Detector._preloaded_model is None:
            import torch_tensorrt
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            base_model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained=True, verbose=False)
            self.class_names = list(base_model.names.values())
            self.model = torch.export.load("/home/andrey/repositories/object_detection_sketch/src/models/trt.pt2").module()
        else:
            self.model = Yolo_v5s_tensorRT_Detector._preloaded_model
            self.class_names = Yolo_v5s_tensorRT_Detector._preloaded_class_names
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print('Загружена yolo с TensorRT')

    def predict(self, frame):
        """Оптимизированный predict"""
        # Подготовка кадра
        image_tensor, pad_info = self._prepare_frame_for_yolo(frame, target_size=640)
        image_tensor = image_tensor.to(self.device)
        
        # Инференс
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Оптимизированная пост-обработка
        return self._postprocess_predictions(predictions[0].cpu(), pad_info)
    
    def _postprocess_predictions(self, predictions, pad_info):
        """Выделенная пост-обработка"""
        # Извлечение данных
        boxes = self._raw_to_xyxy(predictions[:, :4])
        objectness = predictions[:, 4]
        class_probs = predictions[:, 5:]
        
        # Фильтрация по confidence
        max_class_probs, max_class_ids = torch.max(class_probs, dim=-1)
        total_confidence = max_class_probs * objectness
        
        valid = total_confidence > self.score_threshold
        if not valid.any():
            return np.array([]), np.array([]), np.array([]), []
        
        boxes = boxes[valid]
        labels = max_class_ids[valid]
        scores = total_confidence[valid]
        
        # NMS
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        
        boxes = boxes[keep].numpy()
        labels = labels[keep].numpy()
        scores = scores[keep].numpy()
        
        # Коррекция координат
        boxes = self._correct_bbox_coordinates(boxes, pad_info)
        
        class_names = [self.class_names[label] for label in labels]
        return boxes, scores, labels, class_names

    def _prepare_frame_for_yolo(self, frame, target_size=640):
        """Оптимизированная подготовка кадра"""
        # Конвертация цвета
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]
        
        # Letterbox
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # Ресайз (использую INTER_LINEAR для скорости)
        img_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Паддинг
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        img_padded[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = img_resized
        
        # Конвертация в тензор (оптимизировано)
        img_array = np.ascontiguousarray(img_padded.transpose(2, 0, 1))
        tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0).unsqueeze(0)
        
        return tensor, {'scale': scale, 'pad_x': paste_x, 'pad_y': paste_y, 'orig_size': (orig_w, orig_h)}

    @staticmethod
    def _raw_to_xyxy(raw_boxes):
        """Конвертация [cx, cy, w, h] -> [x1, y1, x2, y2]"""
        cx, cy, w, h = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

    @staticmethod
    def _correct_bbox_coordinates(bboxes, pad_info):
        """Коррекция координат bbox"""
        if len(bboxes) == 0:
            return bboxes
        
        scale, pad_x, pad_y = pad_info['scale'], pad_info['pad_x'], pad_info['pad_y']
        orig_w, orig_h = pad_info['orig_size']
        
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_x) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_y) / scale
        
        # Clip
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, orig_w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, orig_h)
        
        return bboxes