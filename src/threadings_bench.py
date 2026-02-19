import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class VideoResult:
    video_path: str
    frame_count: int
    total_time: float
    fps: float
    mean_inference_ms: float


from myUtils.detector_factory import DetectorFactory


def process_single_video(
    model_type: str, 
    device: str, 
    score_threshold: float,
    video_path: Path, 
    classes: List[str],
    thread_id: int,
    preloaded_model=None,
    start_barrier: threading.Barrier = None
) -> Optional[VideoResult]:
    """Обработка одного видео в отдельном потоке"""
    import cv2
    
    print(f"[Поток {thread_id}] СТАРТ создания детектора для {video_path.name} в {time.strftime('%H:%M:%S')}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Поток {thread_id}: Ошибка открытия видео: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"[Поток {thread_id}] Создаю детектор...")
    try:
        detector = DetectorFactory.create_detector(
            model_type=model_type,
            device=device,
            score_threshold=score_threshold,
            preloaded_model=preloaded_model
        )
    except Exception as e:
        print(f"[Поток {thread_id}] ОШИБКА создания детектора: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if start_barrier:
        start_barrier.wait()
    
    print(f"[Поток {thread_id}] Детектор создан, СТАРТ {video_path.name} в {time.strftime('%H:%M:%S')}")
    
    start_time = time.perf_counter()
    
    detector.process_video(
        video_path=video_path,
        show=False,
        classes=classes,
    )
    
    total_time = time.perf_counter() - start_time
    actual_fps = total_frames / total_time if total_time > 0 else 0
    
    print(f"[Поток {thread_id}] Готово {video_path.name} - {actual_fps:.2f} FPS в {time.strftime('%H:%M:%S')}")
    
    return VideoResult(
        video_path=str(video_path),
        frame_count=total_frames,
        total_time=total_time,
        fps=actual_fps,
        mean_inference_ms=0
    )


def run_test(video_paths: List[Path], model_type: str, device: str, 
             score_threshold: float, num_workers: int, classes: List[str]) -> tuple:
    """Запуск теста с указанным количеством потоков"""
    print(f"\n{'='*60}")
    print(f"Тест: {num_workers} поток(ов), видео: {len(video_paths)}")
    print(f"{'='*60}")
    
    preloaded_model = None
    if model_type.lower() == 'yolo_tensorrt':
        print("Предварительная загрузка модели TensorRT...")
        from myUtils.yolo_detector import Yolo_v5s_tensorRT_Detector
        preloaded_model = Yolo_v5s_tensorRT_Detector.preload_model()
        print("Модель предзагружена")
    
    overall_start = time.perf_counter()
    results = []
    
    start_barrier = threading.Barrier(num_workers)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx, video_path in enumerate(video_paths):
            future = executor.submit(
                process_single_video,
                model_type,
                device,
                score_threshold,
                video_path,
                classes,
                idx,
                preloaded_model,
                start_barrier
            )
            futures[future] = video_path
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Ошибка в потоке: {e}")
                import traceback
                traceback.print_exc()
    
    overall_time = time.perf_counter() - overall_start
    total_frames = sum(r.frame_count for r in results)
    overall_fps = total_frames / overall_time if overall_time > 0 else 0
    
    print(f"\nРезультат: {overall_fps:.2f} FPS ({num_workers} потоков, {overall_time:.1f}с)")
    
    return results, overall_fps, overall_time


def main():
    parser = argparse.ArgumentParser(description='Тестирование многопоточности обработки видео')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (cuda или cpu)')
    parser.add_argument('--source', type=str, nargs='+', 
                       default=['data/videos/crowd.mp4', 'data/videos/crowd1.mp4', 'data/videos/crowd2.mp4'],
                       help='Список видео для обработки')
    parser.add_argument('--model', type=str, default='frcnn', 
                       choices=['frcnn', 'frcnn_onnx', 'yolo', 'yolo_tensor', 'yolo_tensorRT'],
                       help='Тип модели')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Порог уверенности')
    parser.add_argument('--workers', type=str, default='1,2,3',
                       help='Количество потоков для тестирования (через запятую). Или одно значение для одиночного теста.')
    parser.add_argument('--single', action='store_true',
                       help='Запустить только один тест (без перебора количества потоков)')
    parser.add_argument('--classes', type=str, default='person',
                       help='Классы для детекции')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    video_paths = [base_dir / Path(p) for p in args.source]
    classes = args.classes.split(',')
    worker_counts = [int(w) for w in args.workers.split(',')]
    
    print(f"Доступные видео:")
    for p in video_paths:
        print(f"  {p} - существует: {p.exists()}")
    
    video_paths = [p for p in video_paths if p.exists()]
    
    if not video_paths:
        print("Нет доступных видео!")
        return
    
    all_results = {}
    
    for num_workers in worker_counts:
        results, overall_fps, total_time = run_test(
            video_paths=video_paths,
            model_type=args.model,
            device=args.device,
            score_threshold=args.threshold,
            num_workers=num_workers,
            classes=classes
        )
        all_results[num_workers] = {'fps': overall_fps, 'time': total_time}
        
        print("Очистка CUDA памяти...")
        try:
            from myUtils.yolo_detector import Yolo_v5s_tensorRT_Detector
            Yolo_v5s_tensorRT_Detector.clear_preloaded()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        import gc
        gc.collect()
    
    print("\n" + "="*60)
    print("СВОДКА")
    print("="*60)
    for workers, data in all_results.items():
        print(f"{workers} поток(ов): {data['fps']:.2f} FPS ({data['time']:.1f}с)")
    print("="*60)


if __name__ == "__main__":
    main()
