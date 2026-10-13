<<<<<<< HEAD
\# YOLO-Based Industrial Steel Defect Detection



A computer vision project that detects and localizes six types of steel surface defects using YOLO and PyTorch, with a Streamlit interface for image-based predictions.



\## Defect Classes



\- Crazing

\- Inclusion

\- Patches

\- Pitted surface

\- Rolled-in scale

\- Scratches



\## Features



\- YOLO-based object detection

\- Bounding-box localization of detected defects

\- Confidence scores for predictions

\- Model evaluation using precision, recall, and mAP

\- Streamlit interface for uploading and analyzing images



\## Technologies



\- Python

\- PyTorch

\- Ultralytics YOLO

\- OpenCV

\- Streamlit

\- Pillow



\## Dataset



The project uses the NEU-DET steel surface defect dataset. The dataset and trained weights are not included in this repository.



Dataset source: \[NEU-DET dataset](https://github.com/)



See the dataset configuration and project instructions before running the scripts.



\## Installation



1\. Clone this repository.

2\. Create and activate a Python virtual environment.

3\. Install dependencies:



&#x20;  `python -m pip install -r requirements.txt`



4\. Obtain the dataset and place it in the expected directory structure.

5\. Update the dataset paths in the configuration file.

6\. Place your trained `best.pt` checkpoint in the expected model directory.



\## Run the Web App



`python -m streamlit run app/app.py`



\## Evaluation



The model's reported test evaluation from this project was:



\- Precision: 75.8%

\- Recall: 69.3%

\- mAP@50: 77.0%

\- mAP@50–95: 44.9%



These metrics describe the recorded evaluation run and do not guarantee performance on new images.



\## Limitations



Predictions may be incorrect, especially for visually similar or subtle defects. This project is intended for learning and portfolio demonstration, not as a replacement for professional industrial quality-control systems.



\## License



Add an appropriate license after confirming the dataset's and other dependencies' license terms.

=======
# industrial-steel-defect-detection
ndustrial-steel-defect-detection
>>>>>>> edc1300280fb51be7c240c0a36aec87db2d65dad
