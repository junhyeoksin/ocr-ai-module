import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_template_quality(original_path, template_path, mask_path):
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
    
    plt.tight_layout()
    plt.savefig('app/dataset/comparison.png')
    plt.close()

if __name__ == "__main__":
    # 이미지 경로
    original_path = 'app/dataset/idcard_original.png'
    template_path = 'app/dataset/idcard_template.png'
    mask_path = 'app/dataset/idcard_mask.png'
    
    # 품질 체크 실행
    check_template_quality(original_path, template_path, mask_path) 