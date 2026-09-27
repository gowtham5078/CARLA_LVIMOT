import cv2
import numpy as np


class BRIEFDescriptorExtractor:
    def __init__(self, bytes=32):
        """
        BRIEF descriptor extractor.

        Parameters
        ----------
        bytes : int
            Descriptor length in bytes.
            32 bytes = 256-bit descriptor.
        """

        self.bytes = bytes

        self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
            bytes=self.bytes
        )

    def compute(self, image, keypoints):
        """
        Compute BRIEF descriptors for FAST keypoints.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR or grayscale image.

        keypoints : list
            Keypoints detected by FAST.

        Returns
        -------
        keypoints : list
            Keypoints for which descriptors were successfully computed.

        descriptors : numpy.ndarray
            Binary BRIEF descriptors.
        """

        if image is None:
            raise ValueError("Input image is None.")

        if len(keypoints) == 0:
            return [], None

        # Convert image to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Compute BRIEF descriptors
        keypoints, descriptors = self.extractor.compute(
            gray,
            keypoints
        )

        return keypoints, descriptors
