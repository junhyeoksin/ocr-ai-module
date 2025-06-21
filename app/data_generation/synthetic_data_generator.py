import cv2
import numpy as np
import logging
import random
import string
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import json

logger = logging.getLogger(__name__)

class SyntheticIDCardDataGenerator:
    """신분증 합성 데이터 생성기"""
    
    def __init__(self, template_path: str, font_path: Optional[str] = None):
        """
        합성 데이터 생성기 초기화
        
        Args:
            template_path: 신분증 템플릿 이미지 경로
            font_path: 사용할 폰트 파일 경로
        """
        self.template_path = Path(template_path)
        self.font_path = Path(font_path) if font_path else None
        self.fake = Faker('ko_KR')  # 한국어 데이터 생성
        
        # 템플릿 이미지 로드
        self.template = self._load_template()
        
        # 신분증 필드 위치 정의 (픽셀 좌표)
        self.field_positions = {
            'name': (150, 120),
            'birth_date': (150, 160),
            'address': (150, 200),
            'issue_date': (150, 240),
            'id_number': (150, 280)
        }
        
        logger.info("합성 데이터 생성기 초기화 완료")
    
    def _load_template(self) -> np.ndarray:
        """템플릿 이미지 로드"""
        if not self.template_path.exists():
            raise FileNotFoundError(f"템플릿 이미지를 찾을 수 없습니다: {self.template_path}")
        
        template = cv2.imread(str(self.template_path))
        if template is None:
            raise ValueError(f"템플릿 이미지를 로드할 수 없습니다: {self.template_path}")
        
        return template
    
    def generate_synthetic_data(self, num_samples: int, output_dir: str) -> List[Dict]:
        """
        합성 데이터 생성
        
        Args:
            num_samples: 생성할 샘플 수
            output_dir: 출력 디렉토리
            
        Returns:
            생성된 데이터 정보 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_data = []
        
        logger.info(f"합성 데이터 생성 시작: {num_samples}개 샘플")
        
        for i in range(num_samples):
            try:
                # 개인정보 생성
                person_data = self._generate_person_data()
                
                # 신분증 이미지 생성
                id_image = self._create_id_card_image(person_data)
                
                # 이미지 저장
                image_filename = f"synthetic_id_{i:06d}.png"
                image_path = output_path / image_filename
                cv2.imwrite(str(image_path), id_image)
                
                # 메타데이터 저장
                sample_info = {
                    'image_path': image_filename,
                    'data': person_data,
                    'bounding_boxes': self._get_text_bounding_boxes(person_data)
                }
                
                generated_data.append(sample_info)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"진행률: {i + 1}/{num_samples}")
                    
            except Exception as e:
                logger.warning(f"샘플 {i} 생성 중 오류: {str(e)}")
                continue
        
        # 메타데이터 JSON 파일 저장
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"합성 데이터 생성 완료: {len(generated_data)}개 성공")
        return generated_data
    
    def _generate_person_data(self) -> Dict[str, str]:
        """개인정보 데이터 생성"""
        return {
            'name': self.fake.name(),
            'birth_date': self.fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%Y.%m.%d'),
            'address': self.fake.address().replace('\n', ' '),
            'issue_date': self.fake.date_between(start_date='-5y', end_date='today').strftime('%Y.%m.%d'),
            'id_number': self._generate_id_number()
        }
    
    def _generate_id_number(self) -> str:
        """주민등록번호 형식의 ID 번호 생성 (가상)"""
        # 앞 6자리: 생년월일
        year = random.randint(40, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        
        # 뒤 7자리: 성별 + 랜덤
        gender = random.choice([1, 2, 3, 4])
        random_digits = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        return f"{year:02d}{month:02d}{day:02d}-{gender}{random_digits}"
    
    def _create_id_card_image(self, person_data: Dict[str, str]) -> np.ndarray:
        """신분증 이미지 생성"""
        # 템플릿 복사
        id_image = self.template.copy()
        
        # OpenCV를 PIL로 변환 (한글 폰트 지원을 위해)
        pil_image = Image.fromarray(cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 폰트 로드
        font = self._get_font()
        
        # 각 필드에 텍스트 추가
        for field, text in person_data.items():
            if field in self.field_positions:
                position = self.field_positions[field]
                
                # 텍스트 색상과 크기 조정
                text_color = (0, 0, 0)  # 검은색
                
                # 주소는 작은 폰트 사용
                if field == 'address':
                    small_font = self._get_font(size=16)
                    draw.text(position, text, font=small_font, fill=text_color)
                else:
                    draw.text(position, text, font=font, fill=text_color)
        
        # PIL을 다시 OpenCV로 변환
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 노이즈 및 변형 추가
        result_image = self._add_variations(result_image)
        
        return result_image
    
    def _get_font(self, size: int = 20) -> ImageFont.FreeTypeFont:
        """폰트 로드"""
        try:
            if self.font_path and self.font_path.exists():
                return ImageFont.truetype(str(self.font_path), size)
            else:
                # 기본 폰트 경로들 시도
                font_paths = [
                    "app/dataset/NanumGothic.ttf",
                    "/System/Library/Fonts/AppleGothic.ttc",  # macOS
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu
                    "C:/Windows/Fonts/malgun.ttf"  # Windows
                ]
                
                for font_path in font_paths:
                    if Path(font_path).exists():
                        return ImageFont.truetype(font_path, size)
                
                # 기본 폰트 사용
                return ImageFont.load_default()
                
        except Exception as e:
            logger.warning(f"폰트 로드 실패: {str(e)}. 기본 폰트 사용")
            return ImageFont.load_default()
    
    def _add_variations(self, image: np.ndarray) -> np.ndarray:
        """이미지에 변형 추가 (노이즈, 블러, 회전 등)"""
        result = image.copy()
        
        # 1. 가우시안 노이즈 추가
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, result.shape).astype(np.uint8)
            result = cv2.add(result, noise)
        
        # 2. 블러 효과
        if random.random() < 0.2:
            kernel_size = random.choice([3, 5])
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
        # 3. 밝기 조정
        if random.random() < 0.4:
            brightness = random.uniform(-30, 30)
            result = cv2.convertScaleAbs(result, alpha=1, beta=brightness)
        
        # 4. 대비 조정
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=0)
        
        # 5. 작은 회전
        if random.random() < 0.2:
            angle = random.uniform(-2, 2)
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, rotation_matrix, (w, h))
        
        return result
    
    def _get_text_bounding_boxes(self, person_data: Dict[str, str]) -> Dict[str, Tuple[int, int, int, int]]:
        """텍스트 바운딩 박스 정보 생성"""
        bounding_boxes = {}
        
        for field, text in person_data.items():
            if field in self.field_positions:
                x, y = self.field_positions[field]
                
                # 텍스트 길이에 따른 대략적인 바운딩 박스 계산
                text_width = len(text) * 12  # 대략적인 문자 폭
                text_height = 25
                
                # 주소는 더 작은 폰트
                if field == 'address':
                    text_width = len(text) * 8
                    text_height = 20
                
                bounding_boxes[field] = (x, y, x + text_width, y + text_height)
        
        return bounding_boxes
    
    def generate_annotation_file(self, generated_data: List[Dict], output_path: str, format_type: str = "yolo") -> None:
        """
        어노테이션 파일 생성
        
        Args:
            generated_data: 생성된 데이터 정보
            output_path: 출력 파일 경로
            format_type: 어노테이션 형식 ("yolo", "coco", "pascal_voc")
        """
        if format_type == "yolo":
            self._generate_yolo_annotations(generated_data, output_path)
        elif format_type == "coco":
            self._generate_coco_annotations(generated_data, output_path)
        else:
            logger.warning(f"지원하지 않는 어노테이션 형식: {format_type}")
    
    def _generate_yolo_annotations(self, generated_data: List[Dict], output_path: str) -> None:
        """YOLO 형식 어노테이션 생성"""
        output_dir = Path(output_path).parent
        
        # 클래스 이름 정의
        class_names = ['name', 'birth_date', 'address', 'issue_date', 'id_number']
        
        # classes.txt 파일 생성
        with open(output_dir / "classes.txt", 'w', encoding='utf-8') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        # 각 이미지에 대한 어노테이션 파일 생성
        for sample in generated_data:
            image_name = Path(sample['image_path']).stem
            annotation_file = output_dir / f"{image_name}.txt"
            
            with open(annotation_file, 'w', encoding='utf-8') as f:
                for field, bbox in sample['bounding_boxes'].items():
                    if field in class_names:
                        class_id = class_names.index(field)
                        
                        # YOLO 형식으로 변환 (정규화된 좌표)
                        img_width, img_height = 800, 500  # 템플릿 크기
                        x1, y1, x2, y2 = bbox
                        
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"YOLO 어노테이션 파일 생성 완료: {len(generated_data)}개")
    
    def _generate_coco_annotations(self, generated_data: List[Dict], output_path: str) -> None:
        """COCO 형식 어노테이션 생성"""
        # COCO 형식 구현 (필요시 확장)
        logger.info("COCO 형식 어노테이션 생성 기능은 추후 구현 예정입니다.") 