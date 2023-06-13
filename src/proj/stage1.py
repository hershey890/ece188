from pathlib import Path
from itertools import repeat
import multiprocessing as mp
from typing import List, Dict, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import unsharp_mask, laplace
from skimage.restoration import wiener, unsupervised_wiener


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


def calc_psnr(truth_img: np.ndarray, test_img: np.ndarray) -> float:
    """Calculate the PSNR between two images or arrays of images"""
    assert truth_img.shape == test_img.shape and truth_img.dtype == test_img.dtype, "calc_psnr assertion error"
    truth_img = sk.img_as_float(truth_img)
    test_img = sk.img_as_float(test_img)
    if truth_img.ndim == 2 or truth_img.ndim == 3 and truth_img.shape[-1] == 3:  # single image
        return peak_signal_noise_ratio(truth_img, test_img)
    elif truth_img.ndim == 3 or truth_img.ndim == 4:  # n-images
        return np.mean([peak_signal_noise_ratio(i1, i2) for i1, i2 in zip(truth_img, test_img)])
    else:
        raise ValueError("Invalid image dimensions.")


def calc_ssim(truth_img: np.ndarray, test_img: np.ndarray) -> float:
    """Calculate the SSIM between two images or arrays of images"""
    assert truth_img.shape == test_img.shape and truth_img.dtype == test_img.dtype, "calc_ssim assertion error"
    truth_img = sk.img_as_float(truth_img)
    test_img = sk.img_as_float(test_img)
    if truth_img.ndim == 2:  # 2 grayscale images
        return structural_similarity(truth_img, test_img)
    elif truth_img.ndim == 3 and truth_img.shape[-1] == 3:  # 2 RGB images
        return structural_similarity(truth_img, test_img, channel_axis=2)
    elif truth_img.ndim == 3:  # n grayscale images
        return np.mean([structural_similarity(i1, i2) for i1, i2 in zip(truth_img, test_img)])
    elif truth_img.ndim == 4:  # n RGB images
        return np.mean(
            [
                structural_similarity(i1, i2, channel_axis=2)
                for i1, i2 in zip(truth_img, test_img)
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
    # Benchmark (PSNR, SSIM): (26.5, 0.65). Currently I get 22.4, 0.76
    # baseline: 21.15, 0.70
    def gaussian_wiener(img, sigma=3.5): # 22.02,, 0.67
        # output_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        # pd = 15
        # output_img = cv2.copyMakeBorder(output_img, pd, pd, pd, pd, cv2.BORDER_REFLECT)
        # h, w = output_img.shape[:2]
        # psf = _gaussian2d(h, w, sigma=sigma)
        # y_channel = 2*(output_img[:,:,0]/255) - 1
        # deconvolved_W = wiener(y_channel, psf, balance=0.05, clip=True)
        # deconvolved_W = 1/2*(deconvolved_W + 1)
        # output_img[:,:,0] = (255*deconvolved_W).astype(np.uint8)
        # output_img = cv2.cvtColor(output_img, cv2.COLOR_YUV2RGB)
        # output_img = output_img[pd:-pd, pd:-pd]
        # output_img = cv2.bilateralFilter(output_img, 5, 50, 50)
        # return output_img

        output_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        pd = 30
        output_img = cv2.copyMakeBorder(output_img, pd, pd, pd, pd, cv2.BORDER_REFLECT)
        h, w = output_img.shape[:2]
        psf = _gaussian2d(h, w, sigma=sigma)
        output_img = 2*(output_img/255) - 1

        output_img[:,:,0] = unsupervised_wiener(output_img[:,:,0], psf, clip=True)[0]
        # output_img[:,:,1] = unsupervised_wiener(output_img[:,:,1], psf, clip=True)[0]
        # output_img[:,:,2] = unsupervised_wiener(output_img[:,:,2], psf, clip=True)[0]

        output_img = (255/2*(output_img + 1)).astype(np.uint8)
        output_img = output_img[pd:-pd, pd:-pd]
        output_img = cv2.cvtColor(output_img, cv2.COLOR_YUV2RGB)
        return output_img

    def gaussian_unsharp_mask(img): # 22.4, 0.76
        return np.uint8(255*unsharp_mask(img/255, radius=3, amount=3, channel_axis=2))
    
    return gaussian_wiener(img)
    # return gaussian_unsharp_mask(img)


def filter_gaussian_noise(img) -> np.ndarray:
    """Filters gaussian noise images."""
    # goal 19.5, 0.60, baseline 12.33, 0.18
    # hits 21.30, 0.63
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 5, 50, 150)
    return img


def filter_motion_blur(img) -> np.ndarray:
    """Filters motion blurred images."""
    return img.copy()


def filter_sp_noise(img) -> np.ndarray:
    """Filters salt and pepper noise images."""
    # goal 26.5, 0.90, baseline 14.49, 0.26
    # hits 26.77, 0.92
    return cv2.medianBlur(img, 3)

def filter_speckle_noise(img) -> np.ndarray:
    """Filters speckle noise images."""
    # goal: 20.0, 0.65, baseline 18.4, 0.60
    # hits 
    img =  cv2.medianBlur(img, 3)


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
    _funcs = {
        "gaussian_blur": filter_gaussian_blur,
        "gaussian_noise": filter_gaussian_noise,
        "motion_blur": filter_motion_blur,
        "sp_noise": filter_sp_noise,
        "speckle_noise": filter_speckle_noise,
    }

    def __init__(self, data_folder: str | Path):
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
        imgs_filtered = []
        for i in range(imgs_truth.shape[0]):
            imgs_filtered.append(func(imgs_test[i]))
        imgs_filtered = np.array(imgs_filtered)
        psnr = calc_psnr(imgs_truth, imgs_filtered)
        ssim = calc_ssim(imgs_truth, imgs_filtered)
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
                    self._funcs[blur_type],
                    blur_type,
                )
            ]
        else:
            n_processes = min(mp.cpu_count(), 5)
            blurs, funcs = self._funcs.keys(), self._funcs.values()
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

    def plot(self, blur_type: str, n_images: int = 1, width: int = 20) -> None:
        if n_images > self.imgs_truth.shape[0]:
            raise ValueError("n_images must be <= number of images in data folder.")
        plt.figure(figsize=(width, n_images*width//2))
        plt.title('Truth on left, filtered on right')
        for i in range(n_images):
            img_filtered = self._funcs[blur_type](self.imgs[blur_type][i])
            img_truth = self.imgs_truth[i]
            plt.subplot(n_images, 2, 2*i+1, title='Image Truth')
            plt.imshow(img_truth)
            plt.subplot(n_images, 2, 2*i+2, title='Image Filtered')
            plt.imshow(img_filtered)
        plt.show()