import os
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Optional

from models.generate_idcard_template import IDCardTemplateGenerator
from preprocess.image_preprocessing import ImagePreprocessor
from utils.image_checker import check_template_quality

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR AI 모듈의 메인 프로세서 클래스"""
    
    def __init__(self, dataset_dir: Optional[str] = None):
        """
        OCR 프로세서 초기화
        
        Args:
            dataset_dir: 데이터셋 디렉토리 경로
        """
        if dataset_dir is None:
            script_dir = Path(__file__).parent
            self.dataset_dir = script_dir / "dataset"
        else:
            self.dataset_dir = Path(dataset_dir)
        
        # 구성 요소 초기화
        self.template_generator = IDCardTemplateGenerator()
        self.preprocessor = ImagePreprocessor()
        
        logger.info(f"OCR 프로세서 초기화 완료 - 데이터셋 디렉토리: {self.dataset_dir}")
    
    def process_template_generation(self, 
                                  original_filename: str = "idcard_original.png",
                                  mask_filename: str = "idcard_mask.png",
                                  output_filename: str = "idcard_template.png") -> bool:
        """
        신분증 템플릿 생성 프로세스
        
        Args:
            original_filename: 원본 이미지 파일명
            mask_filename: 마스크 이미지 파일명
            output_filename: 출력 이미지 파일명
            
        Returns:
            처리 성공 여부
        """
        try:
            logger.info("🚀 템플릿 생성 프로세스 시작")
            
            # 파일 경로 설정
            original_path = self.dataset_dir / original_filename
            mask_path = self.dataset_dir / mask_filename
            output_path = self.dataset_dir / output_filename
            
            # 파일 존재 확인
            self._validate_input_files(original_path, mask_path)
            
            # 템플릿 생성
            template = self.template_generator.generate_template(
                str(original_path), str(mask_path), str(output_path)
            )
            
            # 품질 검증
            quality_score = self._evaluate_template_quality(original_path, output_path, mask_path)
            logger.info(f"📊 템플릿 품질 점수: {quality_score:.3f}")
            
            # 결과 시각화
            self._generate_comparison_image(original_path, output_path, mask_path)
            
            logger.info("✅ 템플릿 생성 프로세스 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 템플릿 생성 중 오류 발생: {str(e)}")
            return False
    
    def process_image_preprocessing(self, 
                                  input_filename: str,
                                  output_filename: Optional[str] = None,
                                  **preprocessing_options) -> bool:
        """
        이미지 전처리 프로세스
        
        Args:
            input_filename: 입력 이미지 파일명
            output_filename: 출력 이미지 파일명 (None이면 자동 생성)
            **preprocessing_options: 전처리 옵션들
            
        Returns:
            처리 성공 여부
        """
        try:
            logger.info("🔧 이미지 전처리 프로세스 시작")
            
            # 파일 경로 설정
            input_path = self.dataset_dir / input_filename
            
            if output_filename is None:
                name_parts = input_filename.split('.')
                output_filename = f"{name_parts[0]}_preprocessed.{name_parts[1]}"
            
            output_path = self.dataset_dir / output_filename
            
            # 파일 존재 확인
            if not input_path.exists():
                raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
            
            # 이미지 전처리
            processed_image = self.preprocessor.preprocess(str(input_path), **preprocessing_options)
            
            # 결과 저장
            if processed_image.dtype == np.float32:
                # 정규화된 이미지의 경우 0-255 범위로 변환
                processed_image = (processed_image * 255).astype(np.uint8)
            
            cv2.imwrite(str(output_path), processed_image)
            
            # 품질 메트릭 계산
            original_image = cv2.imread(str(input_path))
            quality_metrics = self.preprocessor.get_image_quality_metrics(original_image)
            processed_quality_metrics = self.preprocessor.get_image_quality_metrics(processed_image)
            
            logger.info("📊 이미지 품질 비교:")
            for metric, value in quality_metrics.items():
                processed_value = processed_quality_metrics.get(metric, 0)
                logger.info(f"  {metric}: {value:.2f} → {processed_value:.2f}")
            
            logger.info("✅ 이미지 전처리 프로세스 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 이미지 전처리 중 오류 발생: {str(e)}")
            return False
    
    def _validate_input_files(self, *file_paths) -> None:
        """입력 파일들의 유효성 검증"""
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    def _evaluate_template_quality(self, original_path: Path, template_path: Path, mask_path: Path) -> float:
        """템플릿 품질 평가"""
        try:
            original = cv2.imread(str(original_path))
            template = cv2.imread(str(template_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            quality_score = self.template_generator.get_inpainting_quality_score(original, template, mask)
            return quality_score
            
        except Exception as e:
            logger.warning(f"품질 평가 중 오류: {str(e)}")
            return 0.0
    
    def _generate_comparison_image(self, original_path: Path, template_path: Path, mask_path: Path) -> None:
        """비교 이미지 생성"""
        try:
            check_template_quality(str(original_path), str(template_path), str(mask_path))
            logger.info("✅ 비교 이미지 생성 완료")
        except Exception as e:
            logger.warning(f"비교 이미지 생성 중 오류: {str(e)}")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="OCR AI 모듈 - 신분증 인식 시스템")
    
    parser.add_argument("--mode", choices=["template", "preprocess", "all"], 
                       default="template", help="실행 모드 선택")
    parser.add_argument("--dataset-dir", type=str, help="데이터셋 디렉토리 경로")
    parser.add_argument("--original", type=str, default="idcard_original.png",
                       help="원본 이미지 파일명")
    parser.add_argument("--mask", type=str, default="idcard_mask.png",
                       help="마스크 이미지 파일명")
    parser.add_argument("--output", type=str, default="idcard_template.png",
                       help="출력 이미지 파일명")
    
    # 전처리 옵션들
    parser.add_argument("--no-rotation", action="store_true",
                       help="기울기 보정 비활성화")
    parser.add_argument("--no-enhancement", action="store_true",
                       help="이미지 강화 비활성화")
    parser.add_argument("--no-binarization", action="store_true",
                       help="이진화 비활성화")
    parser.add_argument("--no-normalization", action="store_true",
                       help="정규화 비활성화")
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    try:
        # 명령행 인수 파싱
        args = parse_arguments()
        
        logger.info("🎯 OCR AI 모듈 시작")
        logger.info(f"실행 모드: {args.mode}")
        
        # OCR 프로세서 초기화
        processor = OCRProcessor(args.dataset_dir)
        
        success = True
        
        # 모드별 실행
        if args.mode in ["template", "all"]:
            success &= processor.process_template_generation(
                args.original, args.mask, args.output
            )
        
        if args.mode in ["preprocess", "all"]:
            # 전처리 옵션 설정
            preprocessing_options = {
                "apply_rotation_correction": not args.no_rotation,
                "apply_enhancement": not args.no_enhancement,
                "apply_binarization": not args.no_binarization,
                "apply_normalization": not args.no_normalization
            }
            
            success &= processor.process_image_preprocessing(
                args.original, **preprocessing_options
            )
        
        if success:
            logger.info("🎉 모든 프로세스가 성공적으로 완료되었습니다!")
        else:
            logger.error("❌ 일부 프로세스에서 오류가 발생했습니다.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류 발생: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 