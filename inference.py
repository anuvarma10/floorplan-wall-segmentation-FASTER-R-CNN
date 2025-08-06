import os
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Directory paths
input_dir = "/content/floorplan_dataset/valid"  # Folder with validation images
output_dir = "/content/inference_results"
os.makedirs(output_dir, exist_ok=True)

# Metadata
metadata = MetadataCatalog.get("floorplan_valid")

# Loop through images for batch inference
for img_name in os.listdir(input_dir):
    if img_name.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_dir, img_name)
        im = cv2.imread(img_path)

        # Run inference
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        # ✅ Filter predictions by confidence (only > 0.7)
        high_conf_preds = instances[instances.scores > 0.7]

        # Visualization
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(high_conf_preds)

        # Save output
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

print(f"✅ Inference completed! Results saved in: {output_dir}")
