from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from processing import _image_read, _extract_bbox, _extract_face, most_similarity
from tkinter import *
from PIL import ImageTk, Image

import cv2
import tensorflow as tf
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16


# clasify image
def classify(file_path):
    # global label_packed
    image = _image_read(file_path)
    bboxs = _extract_bbox(image, single=False)
    faces = []
    # try:
    for bbox in bboxs:
        face = _extract_face(image, bbox, face_scale_thres=(30, 30))
        # face = face.copy()
        faces.append(face)
        # try: lúc đầu try ở đây
        face_rz = cv2.resize(face, (224, 224))
        # add 3 dim to feed to anther model
        another_face_3d = np.repeat(np.expand_dims(face_rz, axis=2), 3, axis=2)
        another_face_3d = np.expand_dims(another_face_3d, axis=0)
        # print(another_face_3d.shape)
        # Embedding face
        # vec = model.predict(face_tf)
        vec = model.predict(another_face_3d)
        # print(another_vec)
        # Tìm kiếm ảnh gần nhất
        with open('./embedding/embedding.npy', 'rb') as f:
            embedding = np.load(f)
        # labels = ['subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject07', 'subject06', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject09', 'subject09', 'subject08', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject13', 'subject12', 'subject12', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject01', 'subject15', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject02', 'subject01', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject04', 'subject03', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject06', 'subject06', 'subject05', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject07', 'subject06', 'subject06', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject09', 'subject08', 'subject08', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject11', 'subject10', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject14', 'subject13', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject01', 'subject15', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject02', 'subject02', 'subject02', 'subject01', 'subject02', 'subject01', 'subject01', 'subject02', 'subject02', 'subject02', 'subject03', 'subject02', 'subject03', 'subject02', 'subject02', 'subject02', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject04', 'subject04', 'subject04', 'subject04', 'subject03', 'subject04', 'subject05', 'subject05', 'subject04', 'subject04', 'subject04', 'subject05', 'subject04', 'subject04', 'subject04', 'subject06', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject06', 'subject05', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject07', 'subject07', 'subject07', 'subject06', 'subject07', 'subject07', 'subject07', 'subject08', 'subject07', 'subject08', 'subject07', 'subject08', 'subject07', 'subject08', 'subject07', 'subject07', 'subject08',
        #           'subject09', 'subject08', 'subject08', 'subject08', 'subject09', 'subject08', 'subject08', 'subject08', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject10', 'subject09', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject11', 'subject10', 'subject11', 'subject10', 'subject11', 'subject11', 'subject11', 'subject10', 'subject11', 'subject12', 'subject12', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject12', 'subject12', 'subject12', 'subject12', 'subject13', 'subject12', 'subject12', 'subject13', 'subject12', 'subject12', 'subject13', 'subject12', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject14', 'subject13', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject14', 'subject15', 'subject15', 'subject15', 'subject15', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject02', 'subject01', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject03', 'subject02', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject06', 'subject06', 'subject05', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject12', 'subject11', 'subject11', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject13', 'subject13', 'subject12', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject14', 'subject14', 'subject13', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject15', 'subject15', 'subject14', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject01', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject02', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject03', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject04', 'subject05', 'subject04', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject05', 'subject06', 'subject05', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject06', 'subject07', 'subject06', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject07', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject08', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject09', 'subject10', 'subject09', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject10', 'subject11', 'subject10', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject11', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject12', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject13', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject14', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15', 'subject15']
        labels = ['subject09', 'subject11', 'subject15', 'subject07', 'subject13', 'subject06', 'subject11', 'subject11', 'subject08', 'subject15', 'subject01', 'subject03', 'subject05', 'subject01', 'subject09', 'subject09', 'subject06', 'subject08', 'subject08', 'subject06', 'subject13', 'subject07', 'subject07', 'subject02', 'subject04', 'subject14', 'subject02', 'subject09', 'subject15', 'subject04', 'subject13', 'subject13', 'subject14', 'subject03', 'subject10', 'subject10', 'subject07', 'subject09', 'subject06', 'subject04', 'subject14', 'subject06', 'subject12', 'subject03', 'subject11', 'subject05', 'subject08', 'subject10', 'subject13', 'subject03', 'subject04', 'subject09', 'subject04', 'subject01', 'subject02', 'subject13', 'subject10', 'subject03', 'subject11', 'subject03', 'subject05', 'subject12', 'subject09', 'subject11', 'subject09', 'subject11', 'subject10', 'subject03', 'subject02', 'subject02', 'subject01', 'subject01', 'subject04', 'subject14', 'subject13', 'subject15', 'subject06', 'subject08', 'subject01', 'subject06', 'subject14', 'subject13', 'subject11', 'subject03', 'subject10', 'subject02', 'subject14', 'subject08', 'subject09', 'subject08', 'subject02', 'subject09', 'subject14', 'subject01', 'subject03', 'subject05', 'subject09', 'subject15', 'subject02', 'subject15', 'subject11', 'subject13', 'subject15', 'subject10', 'subject02', 'subject14', 'subject10', 'subject09', 'subject02', 'subject02', 'subject08', 'subject06', 'subject13', 'subject04', 'subject06', 'subject06', 'subject08', 'subject04', 'subject13', 'subject04', 'subject10', 'subject06', 'subject15', 'subject15', 'subject07', 'subject13', 'subject08', 'subject11', 'subject06', 'subject01', 'subject13', 'subject13', 'subject10', 'subject09', 'subject01', 'subject09', 'subject15', 'subject10', 'subject11', 'subject02', 'subject07', 'subject13', 'subject12', 'subject05', 'subject11', 'subject10', 'subject14', 'subject05', 'subject08', 'subject13', 'subject01', 'subject03', 'subject15', 'subject14', 'subject09', 'subject12', 'subject10', 'subject01', 'subject11', 'subject05', 'subject02', 'subject01', 'subject05', 'subject01', 'subject15', 'subject14', 'subject05', 'subject07', 'subject12', 'subject07', 'subject12', 'subject07', 'subject11', 'subject02', 'subject07', 'subject01', 'subject15', 'subject05', 'subject10', 'subject08', 'subject13', 'subject04', 'subject08', 'subject02', 'subject03', 'subject02', 'subject07', 'subject12', 'subject12', 'subject04', 'subject07', 'subject07', 'subject01', 'subject07', 'subject02', 'subject12', 'subject13', 'subject05', 'subject10', 'subject12', 'subject07', 'subject07', 'subject08', 'subject02', 'subject07', 'subject12', 'subject06', 'subject01', 'subject09', 'subject11', 'subject03', 'subject12', 'subject04', 'subject01', 'subject07', 'subject07', 'subject05', 'subject07', 'subject05', 'subject15', 'subject15', 'subject11', 'subject15', 'subject01', 'subject11', 'subject09', 'subject02', 'subject01', 'subject13', 'subject02', 'subject08', 'subject02', 'subject02', 'subject15', 'subject05', 'subject02', 'subject09', 'subject10', 'subject04', 'subject08', 'subject10', 'subject09', 'subject04', 'subject10', 'subject14', 'subject07', 'subject07', 'subject12', 'subject14', 'subject03', 'subject06', 'subject15', 'subject08', 'subject11', 'subject09', 'subject03', 'subject02', 'subject11', 'subject04', 'subject15', 'subject05', 'subject06', 'subject03', 'subject06', 'subject13', 'subject13', 'subject02', 'subject02', 'subject08', 'subject12', 'subject06', 'subject14', 'subject09', 'subject07', 'subject15', 'subject05', 'subject03', 'subject10', 'subject04', 'subject06', 'subject07', 'subject06', 'subject14', 'subject14', 'subject08', 'subject11', 'subject15', 'subject14', 'subject04', 'subject13', 'subject08', 'subject01', 'subject11', 'subject06', 'subject13', 'subject03', 'subject06', 'subject07', 'subject06', 'subject06', 'subject09', 'subject11', 'subject07', 'subject07', 'subject07', 'subject02', 'subject15', 'subject13', 'subject07', 'subject12', 'subject04', 'subject09', 'subject04', 'subject03', 'subject15', 'subject08', 'subject04', 'subject10', 'subject14', 'subject12', 'subject15', 'subject14', 'subject15', 'subject10', 'subject15', 'subject10', 'subject10', 'subject15', 'subject12', 'subject01', 'subject04', 'subject10', 'subject12', 'subject03', 'subject06', 'subject10', 'subject01', 'subject05', 'subject01', 'subject11', 'subject12', 'subject07', 'subject03', 'subject03', 'subject04', 'subject10', 'subject13', 'subject04', 'subject10', 'subject08', 'subject15', 'subject15', 'subject14', 'subject05', 'subject12', 'subject12', 'subject12', 'subject02', 'subject12', 'subject04', 'subject01', 'subject05', 'subject01', 'subject06', 'subject03', 'subject05', 'subject01', 'subject08', 'subject13', 'subject12', 'subject07', 'subject14', 'subject15', 'subject13', 'subject05',
                  'subject11', 'subject04', 'subject04', 'subject09', 'subject08', 'subject05', 'subject14', 'subject13', 'subject08', 'subject14', 'subject04', 'subject15', 'subject03', 'subject09', 'subject10', 'subject05', 'subject12', 'subject02', 'subject02', 'subject04', 'subject08', 'subject15', 'subject07', 'subject11', 'subject02', 'subject15', 'subject15', 'subject08', 'subject12', 'subject04', 'subject09', 'subject11', 'subject14', 'subject02', 'subject09', 'subject12', 'subject13', 'subject06', 'subject06', 'subject06', 'subject12', 'subject06', 'subject02', 'subject11', 'subject05', 'subject13', 'subject12', 'subject05', 'subject15', 'subject06', 'subject10', 'subject13', 'subject11', 'subject08', 'subject09', 'subject01', 'subject09', 'subject07', 'subject03', 'subject13', 'subject10', 'subject15', 'subject03', 'subject02', 'subject08', 'subject07', 'subject07', 'subject06', 'subject14', 'subject02', 'subject13', 'subject05', 'subject06', 'subject01', 'subject05', 'subject11', 'subject09', 'subject11', 'subject04', 'subject08', 'subject04', 'subject07', 'subject01', 'subject13', 'subject04', 'subject01', 'subject04', 'subject15', 'subject10', 'subject14', 'subject12', 'subject01', 'subject09', 'subject11', 'subject12', 'subject03', 'subject11', 'subject13', 'subject14', 'subject11', 'subject06', 'subject05', 'subject05', 'subject14', 'subject11', 'subject15', 'subject03', 'subject04', 'subject06', 'subject04', 'subject09', 'subject14', 'subject09', 'subject12', 'subject12', 'subject03', 'subject11', 'subject03', 'subject07', 'subject10', 'subject10', 'subject02', 'subject03', 'subject13', 'subject15', 'subject01', 'subject07', 'subject08', 'subject11', 'subject07', 'subject06', 'subject05', 'subject15', 'subject11', 'subject08', 'subject02', 'subject13', 'subject14', 'subject11', 'subject04', 'subject01', 'subject04', 'subject11', 'subject01', 'subject09', 'subject05', 'subject07', 'subject06', 'subject05', 'subject14', 'subject08', 'subject01', 'subject05', 'subject03', 'subject08', 'subject14', 'subject02', 'subject03', 'subject10', 'subject03', 'subject10', 'subject14', 'subject09', 'subject13', 'subject08', 'subject15', 'subject03', 'subject03', 'subject02', 'subject08', 'subject10', 'subject10', 'subject04', 'subject11', 'subject03', 'subject14', 'subject06', 'subject15', 'subject05', 'subject14', 'subject07', 'subject13', 'subject01', 'subject05', 'subject09', 'subject02', 'subject12', 'subject12', 'subject14', 'subject07', 'subject08', 'subject01', 'subject13', 'subject13', 'subject05', 'subject02', 'subject06', 'subject06', 'subject10', 'subject01', 'subject09', 'subject04', 'subject02', 'subject09', 'subject14', 'subject01', 'subject09', 'subject01', 'subject01', 'subject09', 'subject12', 'subject08', 'subject09', 'subject15', 'subject07', 'subject03', 'subject08', 'subject02', 'subject04', 'subject05', 'subject08', 'subject05', 'subject06', 'subject05', 'subject04', 'subject08', 'subject09', 'subject11', 'subject11', 'subject08', 'subject03', 'subject12', 'subject01', 'subject11', 'subject10', 'subject15', 'subject05', 'subject01', 'subject14', 'subject14', 'subject01', 'subject14', 'subject05', 'subject05', 'subject05', 'subject11', 'subject03', 'subject10', 'subject14', 'subject08', 'subject11', 'subject02', 'subject09', 'subject14', 'subject10', 'subject12', 'subject06', 'subject09', 'subject08', 'subject15', 'subject10', 'subject14', 'subject05', 'subject03', 'subject02', 'subject03', 'subject12', 'subject12', 'subject10', 'subject10', 'subject03', 'subject08', 'subject12', 'subject12', 'subject11', 'subject09', 'subject13', 'subject06', 'subject13', 'subject03', 'subject02', 'subject03', 'subject11', 'subject05', 'subject03', 'subject12', 'subject04', 'subject11', 'subject04', 'subject10', 'subject08', 'subject03', 'subject01', 'subject06', 'subject06', 'subject06', 'subject12', 'subject10', 'subject07', 'subject15', 'subject03', 'subject05', 'subject04', 'subject14', 'subject15', 'subject13', 'subject05', 'subject05', 'subject04', 'subject13', 'subject14', 'subject10', 'subject09', 'subject06', 'subject05', 'subject03', 'subject12', 'subject06', 'subject01', 'subject12', 'subject11', 'subject02', 'subject10', 'subject04', 'subject06', 'subject10', 'subject13', 'subject12', 'subject07', 'subject13', 'subject03', 'subject14', 'subject07', 'subject07', 'subject07', 'subject04', 'subject09', 'subject14', 'subject10', 'subject01', 'subject06', 'subject07', 'subject14', 'subject15', 'subject08', 'subject14', 'subject09', 'subject04', 'subject13', 'subject13', 'subject12', 'subject01', 'subject13', 'subject09', 'subject01', 'subject15', 'subject08', 'subject15', 'subject02', 'subject08', 'subject03', 'subject12', 'subject06', 'subject12', 'subject05', 'subject04', 'subject11', 'subject09', 'subject14', 'subject04', 'subject01', 'subject08', 'subject07', 'subject02', 'subject02']
        name = most_similarity(embedding, vec, labels)
        label.configure(foreground='#011638', text=name)
    # except:
    #     name = "None"
    #     label.configure(foreground='#011638', text=name)

    # Tìm kiếm các bbox
    # (startY, startX, endY, endX) = bbox
    # minX, maxX = min(startX, endX), max(startX, endX)
    # minY, maxY = min(startY, endY), max(startY, endY)
    # pred_proba = accuracy_score(y_predict_list, Y_train)
    # text = "{}: {:.2f}%".format(another_name, pred_proba * 100)
    # y = startY - 10 if startY - 10 > 10 else startY + 10
    # # cv2.rectangle(image, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
    # # cv2.putText(image, text, (minX, y),
    # #   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.rectangle(image, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
    # cv2.putText(image, another_text, (minX, y),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # class_x=numpy.argmax(pred,axis=1)
    # name = classes[int(class_x) + 1]
    # print(name)
    # label.configure(foreground='#011638', text=name)


def another_base_network():
    model = VGG16(include_top=True, weights=None,
                  input_tensor=Input(shape=(224, 224, 3)))
    # for layer in model.layers[:-5]:
    #     layer.trainable = False
    dense = Dense(128)(model.layers[-4].output)
    norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense)
    model = Model(inputs=[model.input], outputs=[norm2])
    return model


# show button
def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156',
                         foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


# upload image from computer
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),
                           (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


# load the trained model to classify sign
# from tensorflow.keras.Model import load_weights

model = another_base_network()
model = load_model('./checkpoint/another_face_reco_YALE.h5',
                   custom_objects={'CustomModel': another_base_network}, compile=False)
# model.load_weights('./checkpoint/another_face_reco_YALE.h5')
# use GPU
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# dictionary to label all traffic signs class.
# classes = {1: 'Subject01',
#            2: 'Subject02',
#            3: 'Subject03',
#            4: 'Subject04',
#            5: 'Subject05',
#            6: 'Subject06',
#            7: 'Subject07',
#            8: 'Subject08',
#            9: 'Subject09',
#            10: 'Subject10',
#            11: 'Subject11',
#            12: 'Subject12',
#            13: 'Subject13',
#            14: 'Subject14',
#            15: 'Subject15',
#            }

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Face Recognition')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

upload = Button(top, text="Upload an image",
                command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white',
                 font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Your face image",
                pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
