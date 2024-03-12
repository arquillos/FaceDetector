"""Face detection script"""
import glob
from typing import Final

import cv2


FRONTAL_FACE_HAAR: Final = r"resources\haarcascade_frontalface_default.xml"
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
WINDOW_PERIOD: Final = 2000
WINDOW_TITLE: Final = "Face detection"


def get_all_images() -> list[str]:
    """Get all the image paths from the folder"""
    images = glob.glob(pathname="resources/images/*.jpg")
    print(images)
    return images


def show_image(image):
    """Show the image for a brief period"""
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW_TITLE,  image)
    cv2.waitKey(WINDOW_PERIOD)
    cv2.destroyWindow(WINDOW_TITLE)


if __name__ == "__main__":
    detector = cv2.CascadeClassifier(FRONTAL_FACE_HAAR)

    for image in get_all_images():
        print(f"Image: {image}")

        # Converting the image to grey
        img = cv2.imread(image)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting faces
        faces = detector.detectMultiScale(image=grey_img, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            final_img = cv2.rectangle(
                img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2
                )
            show_image(final_img)
