
# Object Detection Sketch

Инструмент для детекции объектов на изображениях и видео с поддержкой моделей **Faster R-CNN** и **YOLO**.  
Подходит для обработки видео в реальном времени и пакетной обработки файлов.

Сравнение моделей в **data/output/README.md**

## Возможности

- Поддержка моделей **Faster R-CNN** и **YOLO**
- Работа с изображениями, видео и веб-камерой
- Подсчёт FPS и статистики задержек
- Настраиваемый порог уверенности
- Фильтрация по классам
- Автоматическое использование GPU (CUDA)
- Визуализация bounding box’ов, классов и confidence

---

## Установка

```bash
git clone <repository-url>
cd object_detection_sketch
pip install -r requirements.txt
````

---

## Использование

### CLI (командная строка)

```bash
# YOLO (по умолчанию)
python src/main.py --model yolo --source data/videos/crowd.mp4

# Faster R-CNN с кастомным порогом
python src/main.py --model frcnn --source data/videos/crowd.mp4 --threshold 0.7

# Без отображения окна
python src/main.py --model yolo --source data/videos/crowd.mp4 --no-show

# Детекция только людей
python src/main.py --model yolo --source data/videos/crowd.mp4 --classes "person"

# Принудительное использование CPU
python src/main.py --model frcnn --source data/videos/crowd.mp4 --device cpu
```

---

### Python API

```python
from src.main import Detector

detector = Detector(
    model_type='yolo',
    device='cuda',  # или 'cpu'
    score_threshold=0.5
)

# Детекция изображения
boxes, scores, labels, class_names = detector.predict("image.jpg")

# Визуализация
result = detector.visualize(
    "image.jpg",
    boxes, scores, labels, class_names,
    vis_class=["person", "car"]
)

# Обработка видео
detector.process_video(
    video_path="video.mp4",
    show=True,
    classes=["person"]
)
```

---

## Структура проекта

```
object_detection_sketch/
├── src/
│   ├── main.py            # Детектор и CLI
│   └── models/
│       ├── fasterrcnn_resnet50_fpn.pth
│       └── yolo26n.pt
├── data/
│   ├── videos/            # Входные видео
│   └── output/            # Результаты
│       ├── frcnn.mp4
│       ├── yolo.mp4
│       └── *.json
├── tests/
│   ├── test.ipynb
│   └── sample_images/
├── requirements.txt
└── README.md
```

---

## Модели

### Faster R-CNN (ResNet-50 FPN)

* Двухэтапный детектор
* 80 классов COCO
* Высокая точность
* Подходит, когда важнее качество, чем скорость

### YOLOv8n

* Одноэтапный детектор
* 80 классов COCO
* Высокая скорость, real-time
* Подходит для потокового видео

---

## Параметры CLI

| Аргумент      | Описание                               |
| ------------- | -------------------------------------- |
| `--model`     | `frcnn` или `yolo`                     |
| `--source`    | Путь к видео или изображению           |
| `--threshold` | Порог уверенности                      |
| `--device`    | `cuda`, `cpu` или `auto`               |
| `--no-show`   | Отключить отображение                  |
| `--classes`   | Классы для отображения (через запятую) |

---

## Выходные данные

Файлы сохраняются в `data/output/`:

* `{model}.mp4` — видео с детекциями
* `{model}.json` — количество объектов по классам
* `{model}_{device}_stats.json` — метрики производительности:

  * FPS
  * mean / median / std latency
  * p95 / p99
  * min / max

---

## Рекомендации по производительности

* **YOLO** для real-time
* **Faster R-CNN** для максимальной точности
* Фильтр классов  `--classes`
* `--threshold` для уменьшения ложных срабатываний

---

## Требования

* Python 3.8+
* PyTorch
* TorchVision
* OpenCV
* Ultralytics
* NumPy
* Pillow

