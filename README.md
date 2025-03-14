# OCR AI Module for ID Card Recognition (신분증 인식 OCR AI 모듈)

이 프로젝트는 신분증 및 문서 인식을 위한 OCR(Optical Character Recognition) AI 모듈입니다. 신분증/운전면허증의 텍스트를 정확하게 인식하기 위한 이미지 전처리, 데이터셋 생성, 그리고 딥러닝 기반 OCR 시스템을 구현합니다.

## 🎯 주요 기능

### 1. 이미지 전처리 모듈
- Adaptive Gaussian Thresholding을 통한 이미지 이진화
- 신분증 이미지 품질 개선 및 노이즈 제거
- 기울기 보정 (Affine Transformation)

### 2. 신분증 템플릿 생성
- OpenCV inpainting을 활용한 개인정보 영역 제거
- 고품질 신분증 템플릿 생성
- YOLO 기반 자동 영역 탐지 (개발 예정)

### 3. 가상 데이터셋 생성
- 랜덤 텍스트 정보 생성 및 합성
- TextRecognitionDataGenerator 활용
- 다양한 폰트와 스타일 적용

### 4. OCR 모델 (개발 예정)
- YOLO 기반 신분증 영역 검출
- CRNN 기반 텍스트 인식
- 딥러닝 기반 고성능 OCR 파이프라인

## 🚀 현재 진행 상황

### ✅ 구현 완료
- 이미지 전처리 모듈 기본 구현
- OpenCV 기반 템플릿 생성 기능
- 기본적인 랜덤 텍스트 삽입 기능

### 🔄 진행 중
- 이미지 품질 검증 시스템
- 신분증 기울기 보정 알고리즘
- 가상 데이터셋 생성 파이프라인

### 📋 향후 계획
- YOLO + CRNN 기반 OCR 모델 구현
- 자동 텍스트 정렬 시스템
- 데이터 생성 파이프라인 고도화

## 🔧 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd ocr-ai-module
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 📚 프로젝트 구조
├── app/
│ ├── init.py
│ ├── main.py
│ └── models/
│ ├── init.py
│ └── generate_idcard_template.py
├── dataset/
│ ├── idcard_original.png
│ └── idcard_mask.png
├── requirements.txt
└── README.md

## 🔍 사용 방법

현재는 템플릿 생성 기능만 구현되어 있습니다:

```bash
python -m app.main
```

## 💻 개발 환경 요구사항

- Python 3.7 이상
- OpenCV
- NumPy
- PyInpaint (또는 OpenCV inpainting)
- Tesseract OCR

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📫 문의사항

문제가 발생하거나 기능 개선 제안이 있으시다면 Issues 탭에 등록해 주세요.
