import sys
from typing import Tuple, List, Union, Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import largestinteriorrectangle as lir


def psnr_ssim_rect(orig_img: np.ndarray, rectified_img: np.ndarray) -> Tuple[float, float]:
    # Convert the image to grayscale
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image (black and white)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    grid = thresh.astype(bool)

    rectangle = lir.lir(grid)
    
    (x_min, y_min) = lir.pt1(rectangle)
    (x_max, y_max) = lir.pt2(rectangle)

    # Crop the image using the inscribed rectangle's coordinates
    cropped_img = orig_img[y_min:y_max, x_min:x_max]
    cropped_img_rect = rectified_img[y_min:y_max, x_min:x_max]
    
    psnr_out = peak_signal_noise_ratio(cropped_img, cropped_img_rect)
    ssim_out = structural_similarity(cropped_img, cropped_img_rect, channel_axis=2)

    return (psnr_out, ssim_out)


def find_matches(img1: np.ndarray, img2: np.ndarray) -> Tuple[Tuple[cv2.KeyPoint], Tuple[cv2.KeyPoint], List[cv2.DMatch]]:
    # Open images and detect keypoints
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None) # len 3655, des1.shape 3655, 128
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None) # len 5618
    
    # match keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches


def find_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    ransac_reproj_thresh: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints from image 1
    kp2 : List[cv2.KeyPoint]
        keypoints from image 2
    good_matches : List[cv2.DMatch]
        list of good matches between keypoints from image 1 and image 2
    ransac_reproj_thresh : float
        reprojection threshold for RANSAC

    Returns
    -------
    M : np.ndarray
        3x3 homography matrix
    mask : np.ndarray
        mask indicating which matches are inliers
    """
    min_match_count = 10

    if len(good_matches) <= min_match_count:
        raise ValueError(
            f"Need at least {min_match_count} matches to find homography"
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh
    )

    return M, mask


def warp_image(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    img : np.ndarray
        image to warp
    M : np.ndarray
        3x3 homography matrix

    Returns
    -------
    warped_img : np.ndarray
        warped image
    """
    h, w = img.shape[:2]
    warped_img = cv2.warpPerspective(img, M, (w, h))
    return warped_img