import cv2

def is_blurry(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < 100:
        return True

    return False