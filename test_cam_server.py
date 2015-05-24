from flask import Flask
import cv2

camera = cv2.VideoCapture(0)

app = Flask(__name__)
from app import views
