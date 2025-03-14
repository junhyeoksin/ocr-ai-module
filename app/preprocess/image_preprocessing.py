import cv2
import numpy as np
from skimage import exposure

class ImagePreprocessor:
    def __init__(self):
        self.target_size = (800, 500)  # 신분증 표준 크기
    
    def preprocess(self, image):
        """전체 전처리 파이프라인"""
        try:
            # 1. 기울기 보정
            corrected = self.correct_perspective(image)
            print("기울기 보정 완료")
            
            # 2. 노이즈 제거 및 대비 강화
            enhanced = self.enhance_image(corrected)
            print("이미지 강화 완료")
            
            # 3. 이진화
            binary = self.apply_adaptive_threshold(enhanced)
            print("이진화 완료")
            
            # 4. 정규화
            normalized = self.normalize_image(binary)
            print("정규화 완료")
            
            return normalized
        except Exception as e:
            print(f"에러 발생: {str(e)}")
            raise
    
    def correct_perspective(self, image):
        """신분증 기울기 보정"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angle = self._calculate_rotation_angle(lines)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            return rotated
        return image
    
    def enhance_image(self, image):
        """이미지 대비 강화 및 노이즈 제거"""
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoisingColored(image)
        # 대비 강화
        enhanced = exposure.equalize_adapthist(denoised)
        return (enhanced * 255).astype(np.uint8)
    
    def apply_adaptive_threshold(self, image):
        """적응형 가우시안 임계값 적용"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def normalize_image(self, image):
        """OCR 모델을 위한 이미지 정규화"""
        # 크기 정규화
        resized = cv2.resize(image, self.target_size)
        # 픽셀값 정규화
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def _calculate_rotation_angle(self, lines):
        """회전 각도 계산"""
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle < 45 or angle > 135:  # 수평/수직 선만 고려
                angles.append(angle)
        return np.median(angles) if angles else 0 