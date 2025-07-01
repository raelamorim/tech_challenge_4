import cv2

def get_video_capture(source: str = "0"):
    """
    Retorna um objeto cv2.VideoCapture.
    - "0" usa a webcam
    - outro valor é interpretado como path para arquivo
    """
    if source == "0":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(source)
