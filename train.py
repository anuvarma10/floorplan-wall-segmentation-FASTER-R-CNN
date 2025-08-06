# train.py

import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

def train_model():
    # ✅ Register datasets
    register_coco_instances("floorplan_train", {}, "floorplan_dataset/train/_annotations.coco.json", "floorplan_dataset/train")
    register_coco_instances("floorplan_valid", {}, "floorplan_dataset/valid/_annotations.coco.json", "floorplan_dataset/valid")
    register_coco_instances("floorplan_test", {}, "floorplan_dataset/test/_annotations.coco.json", "floorplan_dataset/test")

    # ✅ Config
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("floorplan_train",)
    cfg.DATASETS.TEST = ("floorplan_valid",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137847281/model_final_280758.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3   # wall, door, window
    cfg.MODEL.DEVICE = "cuda"  # Use GPU

    # ✅ Output Directory
    cfg.OUTPUT_DIR = "./output_fasterrcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ✅ Training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if _name_ == "_main_":
    train_model()
