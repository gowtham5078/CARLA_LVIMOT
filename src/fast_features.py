import cv2
import numpy as np


class FASTFeatureDetector:
    def __init__(self, threshold=20, nonmax_suppression=True):
        """
        FAST corner detector.

        Parameters
        ----------
        threshold : int
            FAST intensity threshold.
        nonmax_suppression : bool
            Removes weaker nearby keypoints.
        """
        self.threshold = threshold
        self.nonmax_suppression = nonmax_suppression

        self.detector = cv2.FastFeatureDetector_create(
            threshold=self.threshold,
            nonmaxSuppression=self.nonmax_suppression
        )

    def detect(self, image):
        """
        Detect FAST keypoints in an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR or grayscale image.

        Returns
        -------
        keypoints : list
            Detected FAST keypoints.
        """

        if image is None:
            raise ValueError("Input image is None.")

        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect FAST keypoints
        keypoints = self.detector.detect(gray, None)

        return keypoints

    def save_visualization(self, image, keypoints, output_file):
        """
        Draw detected FAST keypoints and save visualization.
        """

        visualization = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        cv2.imwrite(output_file, visualization)
