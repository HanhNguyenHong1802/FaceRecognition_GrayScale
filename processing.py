import face_recognition
import cv2
import numpy as np


def _image_read(image_path):
    image = cv2.imread(image_path)
    # img_binary = convert_to_binary(image, thresh=100)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image_gray.shape)
    # gray_image_3band = np.repeat(image_gray, repeats = 3, axis = -1)
    return image_gray


def _extract_bbox(image, single=True):
    bboxs = face_recognition.face_locations(image)
    if len(bboxs) == 0:
        return None
    if single:
        bbox = bboxs[0]
        # bbox = cv2.cvtColor(bbox, cv2.COLOR_RGB2GRAY)
        return bbox
    else:
        return bboxs


def _extract_face(image, bbox, face_scale_thres=(10000, 1000)):
    h, w = image.shape[:2]
    try:
        (startY, startX, endY, endX) = bbox
    except:
        return None
    minX, maxX = min(startX, endX), max(startX, endX)
    minY, maxY = min(startY, endY), max(startY, endY)
    face = image[minY:maxY, minX:maxX].copy()
    # extract the face ROI and grab the ROI dimensions
    (fH, fW) = face.shape[:2]

    # ensure the face width and height are sufficiently large
    if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
        return None
    else:
        cv2.imshow("face", face)
        return face


from sklearn.metrics.pairwise import cosine_similarity


def most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    # print(sim)
    print(labels)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label
