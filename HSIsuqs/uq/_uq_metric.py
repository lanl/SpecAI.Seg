from abc import ABC, abstractmethod


class UQMetric(ABC):
    """Abstraction for SUQ score

    Any implementations of a SUQ score should extend this class
    """

    def __init__(self, img, info):
        """Set image and segment values

        Args:
            img (ndarray or dict): original image, or dict with key 'img' with value of 3D array.
            info (any): any info for the suq score, should just be the segmentation mask.
        """
        if isinstance(img, dict):
            self.img = img['img']
        else:
            self.img = img
        if info is not None:
            self.info = info
        self.score = None
        self._change = True

    @abstractmethod
    def evaluate(self, img=None, info=None):
        """Evaluates the score for all segments

        Args:
            img (ndarray or dict, optional): original image, or dict with key 'img' with value of 3D array. Defaults to None.
            info (any, optional): any info for the suq score, should just be the segmentation mask. Defaults to None.

        Raises:
            ValueError: Need either img and info to both be an ndarray, or both be none.

        Returns:
            tuple: the image, the info
        """
        if img is None and info is None:
            img, info = self.img.copy(), self.info.copy()
        elif (img is None) ^ (info is None):
            raise ValueError(f'Need both img and seg to not be None')
        else:
            self.img = img
            self.info = info
        return img, info

    def get_uq(self):
        """Get UQ for all regions

        Returns:
            depends: depends on the implementation, but should be the UQ scores for all regions
        """
        if self.score is None or self._change:
            self.score = self.evaluate(self.img, self.info)
            self._change = False
        return self.score

    @abstractmethod
    def get_uq_reg(self, *reg, img=None, info=None):
        """Returns UQ score for region

        Args:
            *reg (ints): regions to find UQ score (combines the regions to find score).
            img (ndarray or dict, optional): original image, or dict with key 'img' with value of 3D array. Defaults to None.
            info (any, optional): any info for the suq score, should just be the segmentation mask. Defaults to None.
        """
        ...

    @abstractmethod
    def get_q_score_reg(self, *reg, img=None, info=None):
        """Returns the quality score for region

        Args:
            *reg (ints): regions to find Q score (combines the regions to find score).
            img (ndarray or dict, optional): original image, or dict with key 'img' with value of 3D array. Defaults to None.
            info (any, optional): any info for the suq score, should just be the segmentation mask. Defaults to None.
        """
        ...

    def set_img(self, img):
        """Updates the image

        Args:
            img (ndarray or dict, optional): original image, or dict with key 'img' with value of 3D array. Defaults to None.
        """
        self._change = True
        self.__init__(img, None)

    def set_info(self, info):
        """Updates the info

        Args:
            info (any, optional): any info for the suq score, should just be the segmentation mask. Defaults to None.

        Returns:
            any: returns the info
        """
        self._change = True
        if info is not None:
            self.info = info
        return info
