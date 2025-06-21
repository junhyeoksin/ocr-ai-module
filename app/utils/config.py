import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """이미지 전처리 설정"""
    target_width: int = 800
    target_height: int = 500
    apply_rotation_correction: bool = True
    apply_enhancement: bool = True
    apply_binarization: bool = True
    apply_normalization: bool = True
    angle_threshold: float = 2.0
    denoise: bool = True
    enhance_contrast: bool = True
    adaptive_threshold_method: str = "gaussian"
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2

@dataclass
class TemplateGenerationConfig:
    """템플릿 생성 설정"""
    inpaint_radius: int = 3
    inpaint_method: str = "telea"  # "telea" or "ns"
    quality_threshold: float = 0.5

@dataclass
class DataGenerationConfig:
    """데이터 생성 설정"""
    num_samples: int = 1000
    font_size_range: tuple = (20, 40)
    background_color: tuple = (255, 255, 255)
    text_color: tuple = (0, 0, 0)
    blur_range: tuple = (0, 2)
    noise_level: float = 0.1

@dataclass
class OCRConfig:
    """전체 OCR 시스템 설정"""
    preprocessing: PreprocessingConfig
    template_generation: TemplateGenerationConfig
    data_generation: DataGenerationConfig
    
    # 디렉토리 설정
    dataset_dir: str = "dataset"
    output_dir: str = "output"
    log_dir: str = "logs"
    
    # 로깅 설정
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 성능 설정
    use_gpu: bool = False
    num_workers: int = 4

class ConfigManager:
    """설정 관리자 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        설정 관리자 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> OCRConfig:
        """설정 파일 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 중첩된 설정 객체 생성
                preprocessing_config = PreprocessingConfig(**config_data.get('preprocessing', {}))
                template_config = TemplateGenerationConfig(**config_data.get('template_generation', {}))
                data_config = DataGenerationConfig(**config_data.get('data_generation', {}))
                
                # 기타 설정 추출
                other_config = {k: v for k, v in config_data.items() 
                               if k not in ['preprocessing', 'template_generation', 'data_generation']}
                
                config = OCRConfig(
                    preprocessing=preprocessing_config,
                    template_generation=template_config,
                    data_generation=data_config,
                    **other_config
                )
                
                logger.info(f"설정 파일 로드 완료: {self.config_path}")
                return config
            else:
                logger.info("설정 파일이 없어 기본 설정을 사용합니다.")
                return self._create_default_config()
                
        except Exception as e:
            logger.warning(f"설정 파일 로드 중 오류 발생: {str(e)}. 기본 설정을 사용합니다.")
            return self._create_default_config()
    
    def _create_default_config(self) -> OCRConfig:
        """기본 설정 생성"""
        return OCRConfig(
            preprocessing=PreprocessingConfig(),
            template_generation=TemplateGenerationConfig(),
            data_generation=DataGenerationConfig()
        )
    
    def save_config(self) -> None:
        """현재 설정을 파일로 저장"""
        try:
            config_dict = self._config_to_dict(self.config)
            
            # 설정 디렉토리 생성
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"설정 파일 저장 완료: {self.config_path}")
            
        except Exception as e:
            logger.error(f"설정 파일 저장 중 오류 발생: {str(e)}")
    
    def _config_to_dict(self, config: OCRConfig) -> Dict[str, Any]:
        """설정 객체를 딕셔너리로 변환"""
        return {
            'preprocessing': asdict(config.preprocessing),
            'template_generation': asdict(config.template_generation),
            'data_generation': asdict(config.data_generation),
            'dataset_dir': config.dataset_dir,
            'output_dir': config.output_dir,
            'log_dir': config.log_dir,
            'log_level': config.log_level,
            'log_format': config.log_format,
            'use_gpu': config.use_gpu,
            'num_workers': config.num_workers
        }
    
    def get_config(self) -> OCRConfig:
        """현재 설정 반환"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"설정 업데이트: {key} = {value}")
            else:
                logger.warning(f"알 수 없는 설정 키: {key}")
    
    def reset_to_default(self) -> None:
        """기본 설정으로 재설정"""
        self.config = self._create_default_config()
        logger.info("설정이 기본값으로 재설정되었습니다.")

# 전역 설정 관리자 인스턴스
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """전역 설정 관리자 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> OCRConfig:
    """현재 설정 반환"""
    return get_config_manager().get_config() 