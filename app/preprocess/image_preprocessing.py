import cv2
import numpy as np
from skimage import exposure
import logging
from typing import Tuple, Optional, Union
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (800, 500)):
        """
        이미지 전처리기 초기화
        
        Args:
            target_size: 목표 이미지 크기 (width, height)
        """
        self.target_size = target_size
        logger.info(f"이미지 전처리기 초기화 완료 - 목표 크기: {target_size}")
    
    def preprocess(self, image: Union[np.ndarray, str], 
                  apply_rotation_correction: bool = True,
                  apply_enhancement: bool = True,
                  apply_binarization: bool = True,
                  apply_normalization: bool = True) -> np.ndarray:
        """
        전체 전처리 파이프라인
        
        Args:
            image: 입력 이미지 (numpy array 또는 파일 경로)
            apply_rotation_correction: 기울기 보정 적용 여부
            apply_enhancement: 이미지 강화 적용 여부
            apply_binarization: 이진화 적용 여부
            apply_normalization: 정규화 적용 여부
            
        Returns:
            전처리된 이미지
        """
        try:
            # 이미지 로드
            if isinstance(image, str):
                image = self._load_image(image)
            
            original_shape = image.shape
            logger.info(f"전처리 시작 - 원본 크기: {original_shape}")
            
            processed = image.copy()
            
            # 1. 기울기 보정
            if apply_rotation_correction:
                processed = self.correct_perspective(processed)
                logger.info("✅ 기울기 보정 완료")
            
            # 2. 노이즈 제거 및 대비 강화
            if apply_enhancement:
                processed = self.enhance_image(processed)
                logger.info("✅ 이미지 강화 완료")
            
            # 3. 이진화
            if apply_binarization:
                processed = self.apply_adaptive_threshold(processed)
                logger.info("✅ 이진화 완료")
            
            # 4. 정규화
            if apply_normalization:
                processed = self.normalize_image(processed)
                logger.info("✅ 정규화 완료")
            
            logger.info(f"전처리 완료 - 최종 크기: {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"❌ 전처리 중 오류 발생: {str(e)}")
            raise
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """이미지 파일 로드"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        return image
    
    def correct_perspective(self, image: np.ndarray, 
                          angle_threshold: float = 2.0) -> np.ndarray:
        """
        신분증 기울기 보정
        
        Args:
            image: 입력 이미지
            angle_threshold: 보정을 적용할 최소 각도 (도)
            
        Returns:
            기울기가 보정된 이미지
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 가장자리 검출
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 허프 변환으로 직선 검출
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                angle = self._calculate_rotation_angle(lines)
                
                # 각도가 임계값보다 클 때만 보정 적용
                if abs(angle) > angle_threshold:
                    logger.info(f"기울기 보정 적용: {angle:.2f}도")
                    return self._rotate_image(image, angle)
                else:
                    logger.info(f"기울기가 작아 보정 생략: {angle:.2f}도")
            
            return image
            
        except Exception as e:
            logger.warning(f"기울기 보정 중 오류 발생: {str(e)}")
            return image
    
    def enhance_image(self, image: np.ndarray, 
                     denoise: bool = True,
                     enhance_contrast: bool = True) -> np.ndarray:
        """
        이미지 대비 강화 및 노이즈 제거
        
        Args:
            image: 입력 이미지
            denoise: 노이즈 제거 적용 여부
            enhance_contrast: 대비 강화 적용 여부
            
        Returns:
            강화된 이미지
        """
        try:
            enhanced = image.copy()
            
            # 노이즈 제거
            if denoise:
                if len(image.shape) == 3:
                    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
                else:
                    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            # 대비 강화
            if enhance_contrast:
                # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
                if len(enhanced.shape) == 3:
                    # 컬러 이미지의 경우 LAB 색공간에서 L 채널에만 적용
                    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    # 그레이스케일 이미지
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"이미지 강화 중 오류 발생: {str(e)}")
            return image
    
    def apply_adaptive_threshold(self, image: np.ndarray, 
                               method: str = 'gaussian',
                               block_size: int = 11,
                               C: int = 2) -> np.ndarray:
        """
        적응형 임계값 적용
        
        Args:
            image: 입력 이미지
            method: 임계값 방법 ('gaussian' 또는 'mean')
            block_size: 임계값 계산을 위한 블록 크기
            C: 평균에서 차감할 상수
            
        Returns:
            이진화된 이미지
        """
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 적응형 임계값 방법 선택
            if method == 'gaussian':
                adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            else:
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
            
            # 적응형 임계값 적용
            binary = cv2.adaptiveThreshold(
                gray, 255, adaptive_method,
                cv2.THRESH_BINARY, block_size, C
            )
            
            return binary
            
        except Exception as e:
            logger.warning(f"이진화 중 오류 발생: {str(e)}")
            # 간단한 임계값으로 대체
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary
    
    def normalize_image(self, image: np.ndarray, 
                       resize: bool = True,
                       normalize_pixels: bool = True) -> np.ndarray:
        """
        OCR 모델을 위한 이미지 정규화
        
        Args:
            image: 입력 이미지
            resize: 크기 조정 적용 여부
            normalize_pixels: 픽셀값 정규화 적용 여부
            
        Returns:
            정규화된 이미지
        """
        try:
            normalized = image.copy()
            
            # 크기 정규화
            if resize:
                normalized = cv2.resize(normalized, self.target_size, interpolation=cv2.INTER_AREA)
            
            # 픽셀값 정규화 (0-1 범위)
            if normalize_pixels:
                normalized = normalized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.warning(f"정규화 중 오류 발생: {str(e)}")
            return image
    
    def _calculate_rotation_angle(self, lines: np.ndarray) -> float:
        """
        회전 각도 계산
        
        Args:
            lines: 허프 변환으로 검출된 직선들
            
        Returns:
            계산된 회전 각도 (도)
        """
        angles = []
        
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            
            # 수평선에 가까운 선들만 고려 (-45도 ~ 45도)
            if -45 <= angle <= 45:
                angles.append(angle)
        
        if angles:
            # 중앙값 사용하여 이상치 영향 최소화
            return np.median(angles)
        else:
            return 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        이미지 회전
        
        Args:
            image: 입력 이미지
            angle: 회전 각도 (도)
            
        Returns:
            회전된 이미지
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 회전 변환 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전된 이미지의 새로운 크기 계산
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # 변환 행렬 조정
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # 이미지 회전
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        return rotated
    
    def get_image_quality_metrics(self, image: np.ndarray) -> dict:
        """
        이미지 품질 메트릭 계산
        
        Args:
            image: 입력 이미지
            
        Returns:
            품질 메트릭 딕셔너리
        """
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 품질 메트릭 계산
            metrics = {
                'sharpness': self._calculate_sharpness(gray),
                'contrast': self._calculate_contrast(gray),
                'brightness': self._calculate_brightness(gray),
                'noise_level': self._estimate_noise_level(gray)
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"품질 메트릭 계산 중 오류: {str(e)}")
            return {}
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """라플라시안 분산을 이용한 선명도 계산"""
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    def _calculate_contrast(self, gray_image: np.ndarray) -> float:
        """RMS 대비 계산"""
        return gray_image.std()
    
    def _calculate_brightness(self, gray_image: np.ndarray) -> float:
        """평균 밝기 계산"""
        return gray_image.mean()
    
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """노이즈 레벨 추정"""
        # 가우시안 블러 적용 후 차이로 노이즈 추정
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise = cv2.absdiff(gray_image, blurred)
        return noise.mean() 