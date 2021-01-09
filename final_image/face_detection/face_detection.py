#Participants
#ARAVIND REGONDA
#SWAPANDEEP SANDHU
#JAINIL DESAI

import os
import cv2

# Get the path of the this script
CURRENT_FILE_PATH = os.path.dirname(__file__)

# Load the haar-like features
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(
    CURRENT_FILE_PATH, 'haarcascade_frontalface_default.xml'))
EYE_CASCADE = cv2.CascadeClassifier(
    os.path.join(CURRENT_FILE_PATH, 'haarcascade_eye.xml'))
SMILE_CASCADE = cv2.CascadeClassifier(
    os.path.join(CURRENT_FILE_PATH, 'haarcascade_smile.xml'))


def face_detection(bw_img, orig_img):
    faces = FACE_CASCADE.detectMultiScale(bw_img, 1.3, 5)
    for fx, fy, fw, fh in faces:
        cv2.rectangle(orig_img, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        region_of_interest_bw = bw_img[fy:fy+fh, fx:fx+fw]
        region_of_interest_color = orig_img[fy:fy+fh, fx:fx+fw]
        eyes = EYE_CASCADE.detectMultiScale(region_of_interest_bw, 1.1, 22)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(region_of_interest_color, (ex, ey),
                          (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = SMILE_CASCADE.detectMultiScale(region_of_interest_bw, 1.7, 22)
        for sx, sy, sw, sh in smiles:
            cv2.rectangle(region_of_interest_color, (sx, sy),
                          (sx+sw, sy+sh), (0, 0, 255), 2)
    return orig_img


def make_screenshot(img, counter):
    img_path = os.path.join(CURRENT_FILE_PATH, 'imgs')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cv2.imwrite(os.path.join(
        img_path, 'screenshot-{0}.jpeg'.format(counter)), img)


def start_video_capturing(video_capture):
    screenshot_counter = 0
    while True:
        _, img = video_capture.read()
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canvas = face_detection(bw_img, img)
        cv2.imshow('Video', canvas)
        k = cv2.waitKey(1)
        if k == ord('s'):
            make_screenshot(img, screenshot_counter)
            screenshot_counter += 1
        elif k == ord('q'):
            break

        #elif k == 27:
         #   break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 0 = internal webcam, 1 = external webcam
    VIDEO_CAPTURE = cv2.VideoCapture(0)
    start_video_capturing(VIDEO_CAPTURE)
