from myUtils.frcnn_detecor import FasterRCNNDetector, FasterRCNNDetectorONNX
from myUtils.yolo_detector import Yolo_v5s_Detector, Yolo_v5s_tensor_Detector, Yolo_v5s_tensorRT_Detector


class DetectorFactory:
    """Фабрика для создания детекторов"""
    
    @staticmethod
    def create_detector(model_type='frcnn', device=None, score_threshold=0.5, preloaded_model=None):
        """
        Создание детектора
        
        Args:
            model_type: тип модели ('frcnn' или 'yolo')
            device: устройство для вычислений ('cuda' или 'cpu')
            score_threshold: порог уверенности для детекций
            preloaded_model: предзагруженная модель (для TensorRT)
            
        Returns:
            BaseDetector: экземпляр детектора
        """
        model_type = model_type.lower()
        print(model_type)
        if model_type == 'frcnn':
            return FasterRCNNDetector(device=device, score_threshold=score_threshold)
        elif model_type == 'frcnn_onnx':
            return FasterRCNNDetectorONNX(device=device, score_threshold=score_threshold)
        elif model_type == 'yolo':
            return Yolo_v5s_Detector(device=device, score_threshold=score_threshold)
        elif model_type == 'yolo_tensor':
            return Yolo_v5s_tensor_Detector(device=device, score_threshold=score_threshold)
        elif model_type == 'yolo_tensorrt':
            return Yolo_v5s_tensorRT_Detector(device=device, score_threshold=score_threshold, preloaded_model=preloaded_model)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Поддерживаемые: 'frcnn', 'frcnn_onnx'")
        
        