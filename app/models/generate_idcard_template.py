import os
import cv2
import numpy as np
from pyinpaint import Inpaint

class IDCardTemplateGenerator:
    def generate_template(self, original_path, mask_path, output_path):
        """신분증 템플릿 생성"""
        try:
            print(f"✅ 처리 시작...")
            print(f"원본 이미지: {original_path}")
            print(f"마스크 이미지: {mask_path}")
            
            # 1. 이미지 로드 및 전처리
            original_img = cv2.imread(original_path)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 로드
            
            if original_img is None or mask_img is None:
                raise FileNotFoundError("❌ 이미지 로드 실패")
            
            # 채널 수 확인 및 조정
            if original_img.shape[2] == 4:  # RGBA 이미지인 경우
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
            
            # 임시 파일로 저장
            temp_original = "temp_original.png"
            temp_mask = "temp_mask.png"
            
            cv2.imwrite(temp_original, original_img)
            cv2.imwrite(temp_mask, mask_img)
            
            # 2. Inpainting 수행
            inpainter = Inpaint(temp_original, temp_mask)
            inpainted_img = inpainter()
            print("✅ Inpainting 완료")
            
            # 3. 결과 이미지 후처리
            result_img = inpainted_img * 255.0
            result_img = result_img.astype('uint8')
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # 4. 결과 저장
            cv2.imwrite(output_path, result_img)
            print(f"✅ 결과 저장 완료: {output_path}")
            
            # 5. 임시 파일 삭제
            os.remove(temp_original)
            os.remove(temp_mask)
            
            return result_img
            
        except Exception as e:
            print(f"❌ 에러 발생: {str(e)}")
            # 임시 파일 정리
            if os.path.exists("temp_original.png"):
                os.remove("temp_original.png")
            if os.path.exists("temp_mask.png"):
                os.remove("temp_mask.png")
            raise 