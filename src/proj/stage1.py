from pathlib import Path
from itertools import repeat
import multiprocessing as mp
from typing import List, Dict, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import unsharp_mask
from skimage.restoration import wiener


def load_imgs(path: str | Path, encoding: str = 'rgb') -> np.ndarray:
    """A generator that yields images from a directory.
    
    Parameters
    ----------
    path : str | Path
    encoding: {'rgb', 'yuv'}
        The color encoding of the images. If 'yuv', the images are converted to YUV.
    """
    p = Path(path)
    for f in p.iterdir():
        if f.suffix in (".jpg", ".png", ".jpeg", ".bmp"):
            if encoding == 'rgb':
                yield cv2.imread(str(f))[:, :, ::-1]  # -1: BGR to RGB
            elif encoding == 'yuv':
                yield cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2YUV)
            else:
                raise ValueError("Invalid encoding.")


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate the PSNR between two images or arrays of images"""
    assert img1.shape == img2.shape and img1.dtype == img2.dtype, "calc_psnr assertion error"
    if img1.ndim == 2 or img1.ndim == 3 and img1.shape[-1] == 3:  # single image
        return peak_signal_noise_ratio(img1, img2)
    elif img1.ndim == 3 or img1.ndim == 4:  # n-images
        return np.mean([peak_signal_noise_ratio(i1, i2) for i1, i2 in zip(img1, img2)])
    else:
        raise ValueError("Invalid image dimensions.")


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate the SSIM between two images or arrays of images"""
    assert img1.shape == img2.shape and img1.dtype == img2.dtype, "calc_ssim assertion error"
    if img1.ndim == 2:  # 2 grayscale images
        return structural_similarity(img1, img2)
    elif img1.ndim == 3 and img1.shape[-1] == 3:  # 2 RGB images
        return structural_similarity(img1, img2, channel_axis=2)
    elif img1.ndim == 3:  # n grayscale images
        return np.mean([structural_similarity(i1, i2) for i1, i2 in zip(img1, img2)])
    elif img1.ndim == 4:  # n RGB images
        return np.mean(
            [
                structural_similarity(i1, i2, channel_axis=2)
                for i1, i2 in zip(img1, img2)
            ]
        )
    else:
        raise ValueError("Invalid image dimensions.")


def _gaussian2d(h: int, w: int, sigma: float, a: float = 1.0) -> np.ndarray:
    """Create a 2D Gaussian filter.
    """
    x = np.linspace(-w/2, w/2, w)
    y = np.linspace(-h/2, h/2, h)
    X, Y = np.meshgrid(x, y)
    ret = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return a * ret / np.sum(ret)


def filter_gaussian_blur(img) -> np.ndarray:
    """Filters gaussian blurred images."""
    # Benchmark (PSNR, SSIM): (26.5, 0.65)
    def gaussian_wiener(img, sigma=3.5): # 20.6, 0.6
        output_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        h, w = img.shape[:2]
        psf = _gaussian2d(h, w, sigma=3.5)
        y_channel = 2*(img[:,:,0]/255) - 1
        deconvolved_W = wiener(y_channel, psf, balance=0.05, clip=True)
        deconvolved_W = 1/2*(deconvolved_W + 1)
        output_img[:,:,0] = (255*deconvolved_W).astype(np.uint8)
        return cv2.cvtColor(output_img, cv2.COLOR_YUV2RGB)

    def gaussian_unsharp_mask(img): # 22.4, 0.64
        return np.uint8(255*unsharp_mask(img/255, radius=3, amount=3, channel_axis=2))
    
    return gaussian_unsharp_mask(img)


def filter_gaussian_noise(img) -> np.ndarray:
    """Filters gaussian noise images."""
    return img.copy()

# def motion_blur() -> np.ndarray:
#     """Filters motion blurred images."""
#     return self.imgs["motion_blur"]

# def sp_noise() -> np.ndarray:
#     """Filters salt and pepper noise images."""
#     return self.imgs["sp_noise"]

# def speckle_noise() -> np.ndarray:
#     """Filters speckle noise images."""
#     return self.imgs["speckle_noise"]


class FilterEvaluator:
    """A class that runs and evaluates filtering algorithms.

    Parameters
    ----------
    blur_types: Tuple[str]
        A list of blur types to evaluate.
    imgs: dict
        A dictionary of images, where the keys are the blur types.
    """

    input_type = "input_imgs"
    blur_types = (
        "gaussian_blur",
        "gaussian_noise",
        "motion_blur",
        "sp_noise",
        "speckle_noise",
    )

    def __init__(self, data_folder: str | Path, gaussian_sigma: float = 3.75):
        """
        Parameters
        ----------
        data_folder : str | Path
        gaussian_sigma : float
            emperically determined to be on [3.5, 3.75] with binary search for Gaussian blur
        """
        self._data_folder = data_folder
        self.imgs = {}
        self.imgs_truth = np.array(list(load_imgs(data_folder + "/" + self.input_type)))
        for blur_type in self.blur_types:
            self.imgs[blur_type] = np.array(
                list(load_imgs(data_folder + "/" + blur_type))
            )

    @staticmethod
    def _run(
        imgs_truth: np.ndarray, imgs_test: np.ndarray, func: Callable, blur_type: str
    ) -> Tuple[float, float, float]:
        for i in range(imgs_truth.shape[0]):
            imgs_test[i] = func(imgs_test[i])
        psnr = calc_psnr(imgs_truth, imgs_test)
        ssim = calc_ssim(imgs_truth, imgs_test)
        return psnr, ssim, blur_type

    def run(self, blur_type: str = None) -> Dict:
        """Evaluate all denoising algorithms

        Parameters
        ----------
        blur_type : str
            The type of blur to evaluate. If None, all types are evaluated.
        data_folder : str
            The path to the data folder.

        Returns
        -------
        dict:
            blur_type: (psnr: float, ssim: float)
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
            # "motion_blur": self.motion_blur,
            # "sp_noise": self.sp_noise,
            # "speckle_noise": self.speckle_noise,
        }

        def print_result(blur_type, psnr, ssim, psnr_goal, ssim_goal) -> str:
            if psnr > psnr_goal and ssim > ssim_goal:
                print(f"{blur_type}: PASS")
            else:
                print(f"{blur_type}: FAIL")
                print(f"\tGoal: PSNR={psnr_goal:.2f}, SSIM={ssim_goal:.2f}")
                print(f"\tActual: PSNR={psnr:.2f}, SSIM={ssim:.2f}\n")

        if blur_type is not None:
            psnr_ssim_vals = [
                FilterEvaluator._run(
                    self.imgs_truth,
                    self.imgs[blur_type],
                    funcs[blur_type],
                    blur_type,
                )
            ]
        else:
            n_processes = min(mp.cpu_count(), 5)
            blurs, funcs = funcs.keys(), funcs.values()
            imgs = [self.imgs[blur] for blur in blurs]
            imgs_truth = repeat(self.imgs_truth, n_processes)
            with mp.Pool(n_processes) as p:
                psnr_ssim_vals = p.starmap(
                    FilterEvaluator._run,
                    zip(imgs_truth, imgs, funcs, blurs),
                )

        ret = {}
        print(psnr_ssim_vals)
        for psnr, ssim, blur in psnr_ssim_vals:
            ret[blur] = (psnr, ssim)
            print_result(blur, psnr, ssim, *goals[blur])

        return ret
