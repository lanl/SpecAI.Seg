from abc import ABC, abstractmethod


class UnsupervisedMetric(ABC):
    """Abstract class for unsupervised segmentation evaluation metrics.

    Every unsupervised evaluation metric should extend this class.
    It provides common function signatures to process parameters, and get/find the score.
    """
    @abstractmethod
    def evaluate(self, img=None, seg=None):
        """Calculatest the score for a given image and segmentation

        Args:
            img (ndarray, optional): 3D spectral image. Defaults to None.
            seg (ndarray, optional): 2D segmentation mask. Defaults to None.
        """
        if img is None and seg is None:
            img, seg = self.img.copy(), self.seg.copy()
        elif (img is None) ^ (seg is None):
            raise ValueError(f'Need both img and seg to not be None')
        else:
            self.img = img
            self.seg = seg
        return img, seg

    def get_score(self):
        """Returns the score, calculates it if needed.

        Returns:
            float: the score for the image and segmentation
        """
        if self.score is None:
            self.score = self.evaluate(self.img, self.seg)
        return self.score

    @abstractmethod
    def get_region_score(self, lab=None, img=None, seg=None):
        ...

    def __init__(self, img, seg):
        """Sets up the standard variables

        Args:
            img (ndarray or dict): 3D spectral array image, or dict with 'img' key with 3D array value.
            seg (ndarray): 2D segmentation mask
        """
        if isinstance(img, dict):
            self.img = img['img']
        else:
            self.img = img
        self.seg = seg
        self.score = None
