import cv2

from face_detector import DetectFace
from model import Model

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    still_frame = cv2.imread('face_sample/iqbal.jpg')

    model = Model(model_path='model/siamese_vggface_resnet50_224.h5')

    while True:
        is_frame, raw_frame = cam.read()

        if not is_frame:
            break
        
        frame = raw_frame.copy()

        face_detector = DetectFace(frame).using_haar()
        is_face, face_coords = face_detector.extract()
        if is_face:
            for x1, y1, x2, y2 in face_coords:
                crop_face = raw_frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        is_match = model.is_match(still_frame, crop_face)
        if is_match:
            print('Match !!')
        else:
            print('Not Match !!')

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('Preview', frame)

        key_pressed = cv2.waitKey(1) & 0xff
        if key_pressed == ord('x'):
            break
        elif key_pressed == ord('c'):
            cv2.imwrite('face_sample/iqbal.jpg', crop_face)

    cam.release()
    cv2.destroyAllWindows()
