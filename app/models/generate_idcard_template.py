import os
import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IDCardTemplateGenerator:
    def __init__(self, inpaint_radius: int = 3, inpaint_method: str = 'telea'):
        """
        신분증 템플릿 생성기 초기화
        
        Args:
            inpaint_radius: 인페인팅 반경
            inpaint_method: 인페인팅 방법 ('telea' 또는 'ns')
        """
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = inpaint_method
        self.inpaint_flag = cv2.INPAINT_TELEA if inpaint_method == 'telea' else cv2.INPAINT_NS
        
    def generate_template(self, original_path: str, mask_path: str, output_path: str) -> np.ndarray:
        """
        신분증 템플릿 생성
        
        Args:
            original_path: 원본 이미지 경로
            mask_path: 마스크 이미지 경로
            output_path: 출력 이미지 경로
            
        Returns:
            생성된 템플릿 이미지 (numpy array)
        """
        try:
            logger.info(f"템플릿 생성 시작...")
            logger.info(f"원본 이미지: {original_path}")
            logger.info(f"마스크 이미지: {mask_path}")
            
            # 파일 존재 확인
            self._validate_input_files(original_path, mask_path)
            
            # 이미지 로드 및 전처리
            original = self._load_and_preprocess_image(original_path)
            mask = self._load_and_preprocess_mask(mask_path)
            
            # 이미지 크기 일치 확인
            original, mask = self._ensure_same_size(original, mask)
            
            # 인페인팅 수행
            result = self._perform_inpainting(original, mask)
            
            # 결과 저장
            self._save_result(result, output_path)
            
            logger.info("✅ 템플릿 생성 완료!")
            return result
            
        except Exception as e:
            logger.error(f"❌ 템플릿 생성 중 오류 발생: {str(e)}")
            raise
    
    def _validate_input_files(self, original_path: str, mask_path: str) -> None:
        """입력 파일들의 유효성 검증"""
        if not Path(original_path).exists():
            raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {original_path}")
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"마스크 이미지를 찾을 수 없습니다: {mask_path}")
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """원본 이미지 로드 및 전처리"""
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        logger.info(f"원본 이미지 크기: {image.shape}")
        
        # RGBA -> RGB 변환
        if len(image.shape) == 3 and image.shape[2] == 4:
            logger.info("RGBA -> RGB 변환 중...")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image
    
    def _load_and_preprocess_mask(self, mask_path: str) -> np.ndarray:
        """마스크 이미지 로드 및 전처리"""
        mask = cv2.imread(mask_path)
        if mask is None:
            raise ValueError(f"마스크 이미지를 로드할 수 없습니다: {mask_path}")
        
        logger.info(f"마스크 이미지 크기: {mask.shape}")
        
        # 그레이스케일 변환
        if len(mask.shape) == 3:
            logger.info("마스크를 그레이스케일로 변환 중...")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask
    
    def _ensure_same_size(self, original: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """원본과 마스크 이미지 크기 일치"""
        if original.shape[:2] != mask.shape[:2]:
            logger.warning("이미지와 마스크 크기가 다릅니다. 마스크를 원본 크기로 조정합니다.")
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        return original, mask
    
    def _perform_inpainting(self, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """인페인팅 수행"""
        logger.info(f"인페인팅 시작 (방법: {self.inpaint_method}, 반경: {self.inpaint_radius})")
        
        # 마스크 이진화
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 인페인팅 수행
        result = cv2.inpaint(original, binary_mask, self.inpaint_radius, self.inpaint_flag)
        
        return result
    
    def _save_result(self, result: np.ndarray, output_path: str) -> None:
        """결과 이미지 저장"""
        # 출력 디렉토리 생성
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 저장
        success = cv2.imwrite(output_path, result)
        if not success:
            raise RuntimeError(f"결과 이미지 저장 실패: {output_path}")
        
        logger.info(f"✅ 결과 저장 완료: {output_path}")
    
    def get_inpainting_quality_score(self, original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> float:
        """인페인팅 품질 점수 계산 (0-1 사이)"""
        try:
            # 마스크 영역에서의 원본과 결과 차이 계산
            mask_area = mask > 127
            
            if not np.any(mask_area):
                return 1.0  # 마스크가 없으면 완벽한 점수
            
            # 구조적 유사성 지수 계산 (간단한 버전)
            original_masked = original[mask_area]
            result_masked = result[mask_area]
            
            # 표준편차 기반 품질 점수
            std_original = np.std(original_masked)
            std_result = np.std(result_masked)
            
            # 0-1 사이로 정규화
            quality_score = min(1.0, std_result / (std_original + 1e-8))
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"품질 점수 계산 중 오류: {str(e)}")
            return 0.5  # 기본값 반환

    def visualize_results(self, original, mask, result):
        # 결과 시각화 코드는 그대로 유지
        pass 