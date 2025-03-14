import os
import cv2
import numpy as np
from app.models.generate_idcard_template import IDCardTemplateGenerator
from app.preprocess.image_preprocessing import ImagePreprocessor

def main():
    """메인 실행 함수"""
    try:
        # 1. 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(script_dir, "dataset")
        
        original_path = os.path.join(dataset_dir, "idcard_original.png")
        mask_path = os.path.join(dataset_dir, "idcard_mask.png")
        output_path = os.path.join(dataset_dir, "idcard_template.png")
        
        # 2. 파일 존재 확인
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"❌ 원본 이미지가 없습니다: {original_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"❌ 마스크 이미지가 없습니다: {mask_path}")
        
        # 3. 템플릿 생성
        generator = IDCardTemplateGenerator()
        template = generator.generate_template(original_path, mask_path, output_path)
        
        print("✅ 처리 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        raise

def compare_images(original_path, template_path, mask_path):
    """결과 이미지 비교 및 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        # 이미지 로드
        original = cv2.imread(original_path)
        template = cv2.imread(template_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # BGR to RGB 변환
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        
        # 결과 시각화
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(original_rgb)
        plt.axis('off')
        
        plt.subplot(132)
        plt.title('Generated Template')
        plt.imshow(template_rgb)
        plt.axis('off')
        
        plt.subplot(133)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        # 결과 저장
        comparison_path = os.path.join(os.path.dirname(template_path), "comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        
        print(f"✅ 비교 이미지 저장 완료: {comparison_path}")
        
    except Exception as e:
        print(f"⚠️ 이미지 비교 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 