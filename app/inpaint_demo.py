import os
import cv2
import numpy as np
from pyinpaint import Inpaint

# 현재 디렉토리 기준 상대 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'dataset')

# 원본 신분증 이미지
img_path = os.path.join(data_dir, 'idcard_original.png')
# 마스킹 신분증 이미지
mask_path = os.path.join(data_dir, 'idcard_mask.png')

# 출력 파일 경로
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성
output_path = os.path.join(output_dir, 'idcard_template.png')

# 이미지 로드 및 채널 확인
img = cv2.imread(img_path)
mask_img = cv2.imread(mask_path)

print(f"원본 이미지 경로: {img_path}")
print(f"마스크 이미지 경로: {mask_path}")

# 채널 수 확인 및 조정
if img is None:
    print(f"❌ 오류: 원본 이미지를 로드할 수 없습니다: {img_path}")
    exit(1)
if mask_img is None:
    print(f"❌ 오류: 마스크 이미지를 로드할 수 없습니다: {mask_path}")
    exit(1)

print(f"원본 이미지 형태: {img.shape}")
print(f"마스크 이미지 형태: {mask_img.shape}")

# 마스크 이미지가 3채널인 경우 1채널(그레이스케일)로 변환
if len(mask_img.shape) == 3 and mask_img.shape[2] == 3:
    print("마스크 이미지를 RGB에서 그레이스케일로 변환합니다.")
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    # 변환된 마스크 임시 저장
    temp_mask_path = os.path.join(data_dir, 'idcard_mask_temp.png')
    cv2.imwrite(temp_mask_path, mask_img)
    mask_path = temp_mask_path

try:
    # Inpainting 수행
    print("인페인팅 처리 중...")
    inpaint = Inpaint(img_path, mask_path)
    inpainted_img = inpaint()
    
    # 결과 이미지 처리
    result_img = inpainted_img * 255.0
    result_img = result_img.astype('uint8')
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # 결과 저장
    cv2.imwrite(output_path, result_img)
    print(f"✅ 결과 저장 완료: {output_path}")
    
except Exception as e:
    print(f"❌ pyinpaint 오류 발생: {str(e)}")
    print("OpenCV 인페인팅으로 대체합니다...")
    
    # 마스크 이진화
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    
    # OpenCV 인페인팅 (TELEA 알고리즘)
    result_img = cv2.inpaint(img, binary_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # 결과 저장
    cv2.imwrite(output_path, result_img)
    print(f"✅ 결과 저장 완료: {output_path}")

# 임시 파일 삭제
if os.path.exists(os.path.join(data_dir, 'idcard_mask_temp.png')):
    os.remove(os.path.join(data_dir, 'idcard_mask_temp.png')) 