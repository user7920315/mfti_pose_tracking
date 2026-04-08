
'''
import kagglehub
path = kagglehub.dataset_download("kmader/cmu-mocap")
'''
#print("Path to dataset files:", path) #C:\Users\1\.cache\kagglehub\datasets\kmader\cmu-mocap\versions\4



# Скачайте с Roboflow (есть готовые bounding boxes):

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # бесплатно на roboflow.com
project = rf.workspace().project("ping-pong-ball-detection-sszkb")
dataset = project.version(1).download("yolov8")



