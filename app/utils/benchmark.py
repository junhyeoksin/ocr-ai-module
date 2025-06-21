import time
import psutil
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    throughput: Optional[float] = None  # 처리량 (samples/second)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

class PerformanceBenchmark:
    """성능 벤치마킹 시스템"""
    
    def __init__(self, save_results: bool = True, results_dir: str = "benchmark_results"):
        """
        벤치마크 시스템 초기화
        
        Args:
            save_results: 결과 저장 여부
            results_dir: 결과 저장 디렉토리
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_history: List[Dict[str, Any]] = []
        
        logger.info("성능 벤치마킹 시스템 초기화 완료")
    
    def measure_performance(self, func_name: str = None):
        """
        함수 성능 측정 데코레이터
        
        Args:
            func_name: 함수 이름 (None이면 자동 설정)
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 시작 시간 및 리소스 측정
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
                start_cpu = psutil.cpu_percent(interval=0.1)
                
                try:
                    # 함수 실행
                    result = func(*args, **kwargs)
                    
                    # 종료 시간 및 리소스 측정
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
                    end_cpu = psutil.cpu_percent(interval=0.1)
                    
                    # 메트릭 계산
                    execution_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    cpu_usage = (start_cpu + end_cpu) / 2
                    
                    # GPU 사용량 측정 (nvidia-ml-py 필요)
                    gpu_usage = self._get_gpu_usage()
                    
                    # 메트릭 객체 생성
                    metrics = PerformanceMetrics(
                        execution_time=execution_time,
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_usage,
                        gpu_usage_percent=gpu_usage
                    )
                    
                    # 결과 저장
                    name = func_name or func.__name__
                    self._save_benchmark_result(name, metrics, args, kwargs)
                    
                    logger.info(f"🔍 {name} 성능 측정 완료:")
                    logger.info(f"  실행 시간: {execution_time:.3f}초")
                    logger.info(f"  메모리 사용량: {memory_usage:.2f}MB")
                    logger.info(f"  CPU 사용률: {cpu_usage:.1f}%")
                    if gpu_usage:
                        logger.info(f"  GPU 사용률: {gpu_usage:.1f}%")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"함수 실행 중 오류 발생: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
    
    def benchmark_image_processing(self, processor, image_paths: List[str], 
                                 num_iterations: int = 10) -> Dict[str, Any]:
        """
        이미지 처리 성능 벤치마크
        
        Args:
            processor: 이미지 처리기 객체
            image_paths: 테스트할 이미지 경로 리스트
            num_iterations: 반복 횟수
            
        Returns:
            벤치마크 결과
        """
        logger.info(f"이미지 처리 벤치마크 시작: {len(image_paths)}개 이미지, {num_iterations}회 반복")
        
        all_times = []
        memory_usage = []
        
        for iteration in range(num_iterations):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / 1024 / 1024
            
            for image_path in image_paths:
                try:
                    # 이미지 처리 실행
                    processed = processor.preprocess(image_path)
                except Exception as e:
                    logger.warning(f"이미지 처리 실패: {image_path} - {str(e)}")
                    continue
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024 / 1024
            
            iteration_time = end_time - start_time
            iteration_memory = end_memory - start_memory
            
            all_times.append(iteration_time)
            memory_usage.append(iteration_memory)
            
            logger.info(f"반복 {iteration + 1}/{num_iterations}: {iteration_time:.3f}초")
        
        # 통계 계산
        avg_time = np.mean(all_times)
        std_time = np.std(all_times)
        avg_memory = np.mean(memory_usage)
        throughput = len(image_paths) / avg_time
        
        results = {
            'function': 'image_processing_benchmark',
            'num_images': len(image_paths),
            'num_iterations': num_iterations,
            'avg_execution_time': avg_time,
            'std_execution_time': std_time,
            'avg_memory_usage_mb': avg_memory,
            'throughput_images_per_sec': throughput,
            'all_times': all_times,
            'memory_usage': memory_usage
        }
        
        # 결과 저장
        if self.save_results:
            self._save_detailed_results('image_processing_benchmark', results)
        
        logger.info("📊 이미지 처리 벤치마크 결과:")
        logger.info(f"  평균 실행 시간: {avg_time:.3f}±{std_time:.3f}초")
        logger.info(f"  평균 메모리 사용량: {avg_memory:.2f}MB")
        logger.info(f"  처리량: {throughput:.2f} 이미지/초")
        
        return results
    
    def benchmark_template_generation(self, generator, original_path: str, 
                                    mask_path: str, num_iterations: int = 5) -> Dict[str, Any]:
        """
        템플릿 생성 성능 벤치마크
        
        Args:
            generator: 템플릿 생성기 객체
            original_path: 원본 이미지 경로
            mask_path: 마스크 이미지 경로
            num_iterations: 반복 횟수
            
        Returns:
            벤치마크 결과
        """
        logger.info(f"템플릿 생성 벤치마크 시작: {num_iterations}회 반복")
        
        all_times = []
        quality_scores = []
        
        for iteration in range(num_iterations):
            output_path = f"temp_template_{iteration}.png"
            
            start_time = time.time()
            
            try:
                # 템플릿 생성
                template = generator.generate_template(original_path, mask_path, output_path)
                
                # 품질 점수 계산 (있는 경우)
                if hasattr(generator, 'get_inpainting_quality_score'):
                    import cv2
                    original = cv2.imread(original_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    quality = generator.get_inpainting_quality_score(original, template, mask)
                    quality_scores.append(quality)
                
            except Exception as e:
                logger.warning(f"템플릿 생성 실패: {str(e)}")
                continue
            
            end_time = time.time()
            iteration_time = end_time - start_time
            all_times.append(iteration_time)
            
            # 임시 파일 정리
            temp_path = Path(output_path)
            if temp_path.exists():
                temp_path.unlink()
            
            logger.info(f"반복 {iteration + 1}/{num_iterations}: {iteration_time:.3f}초")
        
        # 통계 계산
        avg_time = np.mean(all_times)
        std_time = np.std(all_times)
        avg_quality = np.mean(quality_scores) if quality_scores else None
        
        results = {
            'function': 'template_generation_benchmark',
            'num_iterations': num_iterations,
            'avg_execution_time': avg_time,
            'std_execution_time': std_time,
            'avg_quality_score': avg_quality,
            'all_times': all_times,
            'quality_scores': quality_scores
        }
        
        # 결과 저장
        if self.save_results:
            self._save_detailed_results('template_generation_benchmark', results)
        
        logger.info("📊 템플릿 생성 벤치마크 결과:")
        logger.info(f"  평균 실행 시간: {avg_time:.3f}±{std_time:.3f}초")
        if avg_quality:
            logger.info(f"  평균 품질 점수: {avg_quality:.3f}")
        
        return results
    
    def _get_gpu_usage(self) -> Optional[float]:
        """GPU 사용량 측정"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return None
    
    def _save_benchmark_result(self, func_name: str, metrics: PerformanceMetrics, 
                             args: tuple, kwargs: dict) -> None:
        """벤치마크 결과 저장"""
        result = {
            'timestamp': time.time(),
            'function_name': func_name,
            'metrics': asdict(metrics),
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        }
        
        self.benchmark_history.append(result)
        
        if self.save_results:
            # 개별 결과 파일 저장
            filename = f"{func_name}_{int(time.time())}.json"
            with open(self.results_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    
    def _save_detailed_results(self, benchmark_name: str, results: Dict[str, Any]) -> None:
        """상세 벤치마크 결과 저장"""
        results['timestamp'] = time.time()
        
        filename = f"{benchmark_name}_{int(time.time())}.json"
        with open(self.results_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"벤치마크 결과 저장: {filename}")
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """
        성능 리포트 생성
        
        Args:
            output_path: 리포트 저장 경로
            
        Returns:
            리포트 파일 경로
        """
        if not self.benchmark_history:
            logger.warning("벤치마크 기록이 없습니다.")
            return ""
        
        # 함수별 성능 통계 계산
        function_stats = {}
        for record in self.benchmark_history:
            func_name = record['function_name']
            metrics = record['metrics']
            
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'execution_times': [],
                    'memory_usage': [],
                    'cpu_usage': []
                }
            
            function_stats[func_name]['execution_times'].append(metrics['execution_time'])
            function_stats[func_name]['memory_usage'].append(metrics['memory_usage_mb'])
            function_stats[func_name]['cpu_usage'].append(metrics['cpu_usage_percent'])
        
        # 리포트 생성
        report_lines = [
            "# OCR AI 모듈 성능 리포트",
            f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"총 벤치마크 기록: {len(self.benchmark_history)}개",
            ""
        ]
        
        for func_name, stats in function_stats.items():
            exec_times = stats['execution_times']
            memory_usage = stats['memory_usage']
            cpu_usage = stats['cpu_usage']
            
            report_lines.extend([
                f"## {func_name}",
                f"- 실행 횟수: {len(exec_times)}회",
                f"- 평균 실행 시간: {np.mean(exec_times):.3f}±{np.std(exec_times):.3f}초",
                f"- 평균 메모리 사용량: {np.mean(memory_usage):.2f}±{np.std(memory_usage):.2f}MB",
                f"- 평균 CPU 사용률: {np.mean(cpu_usage):.1f}±{np.std(cpu_usage):.1f}%",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # 파일 저장
        if output_path is None:
            output_path = self.results_dir / f"performance_report_{int(time.time())}.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"성능 리포트 생성 완료: {output_path}")
        return str(output_path)
    
    def plot_performance_trends(self, metric: str = 'execution_time', 
                              save_plot: bool = True) -> None:
        """
        성능 트렌드 시각화
        
        Args:
            metric: 시각화할 메트릭 ('execution_time', 'memory_usage_mb', 'cpu_usage_percent')
            save_plot: 플롯 저장 여부
        """
        if not self.benchmark_history:
            logger.warning("벤치마크 기록이 없습니다.")
            return
        
        # 함수별 데이터 수집
        function_data = {}
        for record in self.benchmark_history:
            func_name = record['function_name']
            timestamp = record['timestamp']
            metric_value = record['metrics'].get(metric, 0)
            
            if func_name not in function_data:
                function_data[func_name] = {'timestamps': [], 'values': []}
            
            function_data[func_name]['timestamps'].append(timestamp)
            function_data[func_name]['values'].append(metric_value)
        
        # 플롯 생성
        plt.figure(figsize=(12, 8))
        
        for func_name, data in function_data.items():
            timestamps = [time.strftime('%H:%M:%S', time.localtime(ts)) for ts in data['timestamps']]
            plt.plot(timestamps, data['values'], marker='o', label=func_name)
        
        plt.title(f'성능 트렌드: {metric}')
        plt.xlabel('시간')
        plt.ylabel(metric)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.results_dir / f"performance_trend_{metric}_{int(time.time())}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"성능 트렌드 플롯 저장: {plot_path}")
        
        plt.show()
    
    def clear_history(self) -> None:
        """벤치마크 기록 초기화"""
        self.benchmark_history.clear()
        logger.info("벤치마크 기록이 초기화되었습니다.")

# 전역 벤치마크 인스턴스
_benchmark = None

def get_benchmark() -> PerformanceBenchmark:
    """전역 벤치마크 인스턴스 반환"""
    global _benchmark
    if _benchmark is None:
        _benchmark = PerformanceBenchmark()
    return _benchmark

def benchmark(func_name: str = None):
    """성능 측정 데코레이터 (편의 함수)"""
    return get_benchmark().measure_performance(func_name) 