from utils.frcnn_detecor import FasterRCNNDetector, FasterRCNNDetectorONNX

class DetectorFactory:
    """Фабрика для создания детекторов"""
    
    @staticmethod
    def create_detector(model_type='frcnn', device=None, score_threshold=0.5):
        """
        Создание детектора
        
        Args:
            model_type: тип модели ('frcnn' или 'yolo')
            device: устройство для вычислений ('cuda' или 'cpu')
            score_threshold: порог уверенности для детекций
            
        Returns:
            BaseDetector: экземпляр детектора
        """
        model_type = model_type.lower()
        print(model_type)
        if model_type == 'frcnn':
            return FasterRCNNDetector(device=device, score_threshold=score_threshold)
        elif model_type == 'frcnn_onnx':
            return FasterRCNNDetectorONNX(device=device, score_threshold=score_threshold)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Поддерживаемые: 'frcnn', 'frcnn_onnx'")
        
        