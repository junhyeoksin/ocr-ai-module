import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # macOS 기본 한글 폰트
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'AppleGothic'  # macOS용 폰트 설정

# 간단한 테스트 이미지 생성
test_image = np.zeros((300, 300, 3), dtype=np.uint8)
test_image[:100, :100] = [255, 0, 0]  # 빨간색 사각형
test_image[100:200, 100:200] = [0, 255, 0]  # 녹색 사각형
test_image[200:300, 200:300] = [0, 0, 255]  # 파란색 사각형

# 이미지 표시
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.title('테스트 이미지', fontproperties=font_prop)
plt.savefig('test_output.png')
print("테스트 이미지가 생성되었습니다. 'test_output.png' 파일을 확인하세요.") 