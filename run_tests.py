#!/usr/bin/env python3
"""Скрипт для запуска тестов всех YOLO моделей с разным количеством потоков"""

import subprocess
import sys
from pathlib import Path

MODELS = ['yolo', 'yolo_tensor', 'yolo_tensorRT'] # 
WORKERS = list(range(8, 0, -1))
VIDEO_SOURCES = []

def get_video_sources(workers: int) -> list:
    """Генерация списка видео источников на основе количества потоков"""
    return [f'data/videos/crowd{i}.mp4' for i in range(1, workers + 1)]

def run_test(model: str, workers: int) -> dict:
    """Запуск одного теста"""
    video_sources = get_video_sources(workers)
    cmd = [
        sys.executable, 'src/threadings_bench.py',
        '--model', model,
        '--device', 'cuda',
        '--workers', str(workers),
        '--source'
    ] + video_sources
    
    print(f"\n{'='*60}")
    print(f"ЗАПУСК: model={model}, workers={workers}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return {'model': model, 'workers': workers, 'returncode': result.returncode}


def main():
    results = []
    
    for model in MODELS:
        print(f"\n{'#'*60}")
        print(f"# ТЕСТИРОВАНИЕ МОДЕЛИ: {model}")
        print(f"{'#'*60}")
        
        for workers in WORKERS:
            result = run_test(model, workers)
            results.append(result)
    
    print("\n" + "="*60)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("="*60)
    print(f"{'Модель':<20} {'Потоки':<8} {'Статус':<10}")
    print("-"*40)
    for r in results:
        status = "OK" if r['returncode'] == 0 else "FAIL"
        print(f"{r['model']:<20} {r['workers']:<8} {status:<10}")


if __name__ == "__main__":
    main()
