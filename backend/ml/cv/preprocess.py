"""Single source of truth for image preprocessing.

Both the dataset-preparation step (training) and the API (serving) call
`load_and_prepare`, so the model always sees images prepared the same way.
Pixel scaling to [-1, 1] lives *inside* the Keras model (a Rescaling layer),
so this module only outputs uint8 RGB arrays.
"""

import cv2
import numpy as np

IMG_SIZE = 224
MIN_SIDE = 64          # reject thumbnails / broken images
BLUR_THRESHOLD = 15.0  # variance of Laplacian below this is "unusably blurry"


class ImageRejected(ValueError):
    pass


def decode(img_bytes: bytes) -> np.ndarray:
    """Decode bytes to an RGB uint8 array (EXIF orientation is applied by OpenCV)."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageRejected("could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return img[top : top + side, left : left + side]


def blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def prepare(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Centre-crop to square, resize with area interpolation, return uint8 RGB."""
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise ImageRejected(f"image too small ({w}x{h})")
    img = center_crop_square(img)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_and_prepare(img_bytes: bytes, size: int = IMG_SIZE) -> np.ndarray:
    return prepare(decode(img_bytes), size)


def average_hash(img: np.ndarray, hash_size: int = 8) -> int:
    """64-bit perceptual hash used to drop near-duplicate training images."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


NEAR_DUP_BITS = 6  # aHashes within this Hamming distance are treated as the same image


def is_near_duplicate(h: int, kept: "np.ndarray", max_bits: int = NEAR_DUP_BITS) -> bool:
    """True if hash `h` is within `max_bits` of any hash in `kept` (uint64 array)."""
    if kept.size == 0:
        return False
    return bool((np.bitwise_count(kept ^ np.uint64(h)) <= max_bits).any())


def variant_hashes(img: np.ndarray) -> list[int]:
    """aHashes of the 8 flip/rotation variants, so augmented copies that were mirrored or
    rotated by 90 degrees are still recognised as duplicates."""
    out = []
    for k in range(4):
        r = np.rot90(img, k)
        out.append(average_hash(np.ascontiguousarray(r)))
        out.append(average_hash(np.ascontiguousarray(r[:, ::-1])))
    return out
