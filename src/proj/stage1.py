from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_imgs(path: str | Path) -> np.ndarray:
    """A generator that yields images from a directory."""
    p = Path(path)
    for f in p.iterdir():
        if f.suffix in (".jpg", ".png", ".jpeg", ".bmp"):
            yield cv2.imread(str(f))[:, :, ::-1]  # -1: BGR to RGB


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate the PSNR between two images or arrays of images"""
    assert img1.shape == img2.shape and img1.dtype == img2.dtype
    if img1.ndim == 2 or \
        img1.ndim == 3 and img1.shape[-1] == 3:  # single image
        return peak_signal_noise_ratio(img1, img2)
    elif img1.ndim == 3 or img1.ndim == 4:  # n-images
        return np.mean(
            [peak_signal_noise_ratio(i1, i2) for i1, i2 in zip(img1, img2)]
        )
    else:
        raise ValueError("Invalid image dimensions.")


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate the SSIM between two images or arrays of images"""
    assert img1.shape == img2.shape and img1.dtype == img2.dtype
    if img1.ndim == 2:  # 2 grayscale images
        return structural_similarity(img1, img2)
    elif img1.ndim == 3 and img1.shape[-1] == 3:  # 2 RGB images
        return structural_similarity(img1, img2, channel_axis=2)
    elif img1.ndim == 3:  # n grayscale images
        return np.mean(
            [structural_similarity(i1, i2) for i1, i2 in zip(img1, img2)]
        )
    elif img1.ndim == 4:  # n RGB images
        return np.mean(
            [
                structural_similarity(i1, i2, channel_axis=2)
                for i1, i2 in zip(img1, img2)
            ]
        )
    else:
        raise ValueError("Invalid image dimensions.")


def filter_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Inverse Gaussian blur on an image."""
    return img


def filter_gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Inverse Gaussian noise on an image."""
    return img


def filter_motion_blur(img: np.ndarray) -> np.ndarray:
    """Inverse motion blur on an image."""
    return img


def filter_salt_pepper_noise(img: np.ndarray) -> np.ndarray:
    """Inverse salt and pepper noise on an image."""
    return img


def filter_speckle_noise(img: np.ndarray) -> np.ndarray:
    """Inverse speckle noise on an image."""
    return img


class Filters:
    """A class that runs and evaluates filtering algorithms.

    Parameters
    ----------
    blur_types: Tuple[str]
        A list of blur types to evaluate.
    imgs: dict
        A dictionary of images, where the keys are the blur types.
    """

    blur_types = (
        "input_imgs",
        "gaussian_blur",
        "gaussian_noise",
        "motion_blur",
        "sp_noise",
        "speckle_noise",
    )

    def __init__(self, data_folder: str | Path):
        self.imgs = {}
        for blur_type in self.blur_types:
            self.imgs[blur_type] = np.array(load_imgs(data_folder + blur_type))

    def gaussian_blur(self) -> np.ndarray:
        """Filters gaussian blurred images."""
        return filter_gaussian_blur(self.imgs["gaussian_blur"])

    def gaussian_noise(self) -> np.ndarray:
        """Filters gaussian noise images."""
        return filter_gaussian_noise(self.imgs["gaussian_noise"])

    def motion_blur(self) -> np.ndarray:
        """Filters motion blurred images."""
        return filter_motion_blur(self.imgs["motion_blur"])

    def sp_noise(self) -> np.ndarray:
        """Filters salt and pepper noise images."""
        return filter_salt_pepper_noise(self.imgs["sp_noise"])

    def speckle_noise(self) -> np.ndarray:
        """Filters speckle noise images."""
        return filter_speckle_noise(self.imgs["speckle_noise"])


def evaluate(data_folder: str | Path, blur_type: str = None) -> None:
    """Evaluate all denoising algorithms

    Parameters
    ----------
    blur_type : str
        The type of blur to evaluate. If None, all types are evaluated.
    data_folder : str
        The path to the data folder.
    """

    goals = {
        "gaussian_blur": (26.5, 0.65),  # psnr, ssim
        "gaussian_noise": (19.5, 0.60),
        "motion_blur": (27.5, 0.70),
        "sp_noise": (26.5, 0.90),
        "speckle_noise": (20.0, 0.65),
    }
    funcs = {
        "gaussian_blur": filter_gaussian_blur,
        "gaussian_noise": filter_gaussian_noise,
        "motion_blur": filter_motion_blur,
        "sp_noise": filter_salt_pepper_noise,
        "speckle_noise": filter_speckle_noise,
    }

    def eval(img1, img2):
        return calc_psnr(img1, img2), calc_ssim(img1, img2)

    def print_result(blur_type, psnr, ssim, psnr_goal, ssim_goal):
        if psnr > psnr_goal and ssim > ssim_goal:
            print(f"{blur_type}: PASS")
        else:
            print(f"{blur_type}: FAIL")
            print(f"\tGoal: PSNR={psnr_goal:.2f}, SSIM={ssim_goal:.2f}")
            print(f"\tActual: PSNR={psnr:.2f}, SSIM={ssim:.2f}")

    imgs_truth = list(load_imgs(data_folder + "input_imgs"))
    if blur_type is None:
        blur_types = [
            "gaussian_blur",
            "gaussian_noise",
            "motion_blur",
            "sp_noise",
            "speckle_noise",
        ]
    else:
        blur_types = [blur_type]

    for blur_type in blur_types:
        imgs_test = list(load_imgs(data_folder + blur_type))
        psnr = 0
        ssim = 0
        for img_truth, img_test in zip(imgs_truth, imgs_test):
            img_test = funcs[blur_type](img_test)
            a, b = eval(img_truth, img_test)
            psnr += a
            ssim += b
        psnr /= len(imgs_truth)
        ssim /= len(imgs_truth)

        psnr_goal, ssim_goal = goals[blur_type]
        print_result(blur_type, psnr, ssim, psnr_goal, ssim_goal)
