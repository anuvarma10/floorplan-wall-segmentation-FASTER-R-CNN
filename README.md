Floorplan to wall Detection Project

This project is aimed at detecting and identifying key structural elements in floorplans includes  walls  using Faster R-CNN (Detectron2). The model is trained on a custom dataset in COCO format and optimized for floorplan image analysis.


---

📌 Project Overview

Understanding floorplans is a crucial part of real estate, interior design, and construction workflows. Automating the detection of structural elements can reduce manual effort and improve accuracy in digital processing of architectural plans.

Key tasks performed in this project:

Data preparation and annotation in COCO format.

Training a Faster R-CNN model on custom floorplan images.

Inference to visualize predictions on test images.

Evaluation using COCO metrics to measure model performance.





🏗 Tech Stack

Framework: Detectron2

Language: Python 3.11

Libraries: PyTorch, OpenCV, Matplotlib, pycocotools

Environment: Google Colab (GPU-accelerated training)



---

📊 Results

The model shows good accuracy in detecting walls.
Visualized predictions confirm reliable wall detection on test samples.




🔑 Key Learnings

Custom dataset preparation in COCO format.

Training object detection models on domain-specific datasets.

Evaluation of model performance using COCO Average Precision metrics.

Performing inference and visualizing results effectively.
