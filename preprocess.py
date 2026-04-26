import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def denoise(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2 or img.shape[2] == 1:
        return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); correct to [-45, 45)
    if angle < -45:
        angle = 90 + angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def preprocess(
    img: Image.Image,
    apply_denoise: bool = True,
    apply_deskew: bool = False,
    apply_sharpen: bool = False,
) -> Image.Image:
    cv_img = pil_to_cv2(img)
    if apply_denoise:
        cv_img = denoise(cv_img)
    if apply_deskew:
        cv_img = deskew(cv_img)
    if apply_sharpen:
        cv_img = sharpen(cv_img)
    return cv2_to_pil(cv_img)
