import cv2

class DetectFace:
    def __init__(self, image):
        self.image = image
        self.face_coords = None

    def using_haar(self):
        face_cascade = cv2.CascadeClassifier('cascadefile/face-detect.xml')

        image = self.image.copy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_coords = face_cascade.detectMultiScale(gray, 1.3, 3)
        self.face_coords = face_coords

        return self

    def using_cnn(self):
        # code here
        return self

    @staticmethod
    def _standardize_coords(coords):
        x1, y1, w, h = coords
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2
    
    def extract(self):
        is_face = False
        face_coords = None

        if len(self.face_coords) > 0:
            is_face = True
            face_coords = [self._standardize_coords(coord) for coord in self.face_coords]

        return is_face, face_coords