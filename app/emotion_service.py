from deepface import DeepFace
from collections import Counter

def detect_emotion(cropped_face):
    try:
        result = DeepFace.analyze(cropped_face, actions=["emotion"], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return None