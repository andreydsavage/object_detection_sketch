
import torch
import torchvision
import onnxruntime

try:
    from utils.base_detector import BaseDetector
except ImportError:
    # Альтернативный вариант
    from .base_detector import BaseDetector

class FasterRCNNDetector(BaseDetector):  
    def _init_model(self):
        self.model_type = 'frcnn'
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
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
        self.model.to(self.device)
        self.model.eval()
        print('Загружена FRCNN')

    def predict(self, frame):
        # Преобразуем в нужный формат
        image_tensor = torchvision.transforms.functional.to_tensor(frame)
        image_tensor = image_tensor.to(self.device)

        # Выполняем предсказание
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Извлекаем результаты
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        boxes, scores, labels, class_names = self._filter_results(boxes, scores, labels)
        return boxes, scores, labels, class_names

import os

class FasterRCNNDetectorONNX(BaseDetector):  
    def _init_model(self):
        self.model_type = 'frcnn_onnx'
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
        if self.device.type == 'cuda':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        print(os.getcwd())
        self.session = onnxruntime.InferenceSession('src/models/frcnn.onnx', providers = providers)
        self.input_name = self.session.get_inputs()[0].name  

        print('Загружена FRCNN_onnx')

    def predict(self, frame):
        # Преобразуем в нужный формат
        image_tensor = torchvision.transforms.functional.to_tensor(frame)
        image_tensor = image_tensor.numpy(force=True)
        # image_tensor = image_tensor.to(self.device)

        boxes, labels, scores = self.session.run([],{self.input_name : [image_tensor]})

        boxes, scores, labels, class_names = self._filter_results(boxes, scores, labels)
        return boxes, scores, labels, class_names