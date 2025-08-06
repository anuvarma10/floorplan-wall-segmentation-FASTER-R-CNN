# model.py

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def load_model(weights_path="./output_fasterrcnn/model_final.pth"):
    """
    Loads the Faster R-CNN model with custom configuration and trained weights.
    """
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = weights_path  # Path to trained model weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # wall, door, window
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Prediction threshold
    cfg.MODEL.DEVICE = "cuda"  # Use GPU (change to "cpu" if needed)
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg
