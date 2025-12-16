# evaluate.py
# 학습시킨 모델 가중치를 불러와서 추론 후 mAP50, mAP50-95, recall, precision을 출력

# train_split, val_split 폴더는 YOLO 추론을 위해 있는 dummy 폴더입니다.
# test 폴더만 올려서 추론 코드 실행하시면 됩니다.

import logging
from ultralytics import YOLO
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("evaluate.log"),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    
    logging.info("===== Evaluation =====")

    model_path = "./model/weights/best.pt"
    data_yaml = "./dataset.yaml"
    
    logging.info(f"사용 모델: {model_path}")
    logging.info(f"사용 데이터셋: {data_yaml}")

    if not os.path.exists(model_path):
        logging.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return
        
    try:
        model = YOLO(model_path)
        logging.info("모델 로드 성공.")
    except Exception as e:
        logging.error(f"모델 로드 중 오류 발생: {e}")
        return 

    logging.info("Test set 평가 시작 (model.val)...")
    results = model.val(
        data=data_yaml,
        split="test",    
        imgsz=640,
        batch=32,
        conf=0.001,
        project="runs/test",
        name="test_results"
    )
    logging.info("평가 완료.")

    logging.info("========== 최종 평가 결과 (Test Set) ==========")
    logging.info(f"전체 mAP50:     {results.box.map50:.4f}")
    logging.info(f"전체 mAP50-95:  {results.box.map:.4f}")
    logging.info(f"Precision: {results.box.mp:.4f}")
    logging.info(f"Recall:    {results.box.mr:.4f}")
    logging.info("==========================================")

if __name__ == '__main__':
    main()