import cv2
import numpy as np
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.txt_label_encoder import load_circle_coord


def crop_coin(img_path, label_path, save_path: str = None, padding: int = 0, show: bool = False):
    """
    Given an image and its segmentation labels, crop every coin out.

        - Note: Does not return value or orientation yet. This will have to be a later addition.

    :param img_path: str - path to image.
        - e.g. '../data/label/img_3141.jpg'
        - Assumed valid path.
    :param label_path: str - path to label.
        - e.g. '../data/label/label_3141.txt'
        - Assumed valid path.
    :param save_path: str - path to save image.
        - e.g. '../data/label_save/'
        - Note: Does not check for missing trailing '/'.
            - If it did, it would fail for trailing '_' or other intentional punctuation.
        - Values:
            - None = save in img_path. e.g. '../data/label/img_3141_'
            - not str = do not save
            - '' = save in current folder. i.e. where you are running the function
            - str = save in that path
    :param padding: amount of extra padding to add
    :param show: bool - whether to show the image or not

    :return: list of crops
    """

    # Save path
    if save_path is None:
        # Copy img_path, replace final '.xxx' with '_'
        cut = img_path.rfind('.')
        save_path = img_path[:cut] + '_'
    elif not isinstance(save_path, str):
        # Do not save if non-string
        save_path = False

    # Load labels
    labels = load_circle_coord(label_path)

    # Open image and create mask
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Draw white circles on (black) mask
    for i, label in enumerate(labels):
        x, y, r = label[:3]

        # Draw white circle
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Overlay mask and img
    coins = img & mask

    # Cut out circles and save crops
    crops = []
    for i, label in enumerate(labels):
        x, y, r = label

        # Cut out square
        crop = coins[y-r-padding:y+r+padding, x-r-padding:x+r+padding]
        crop = ResizeWithAspectRatio(crop, 600)

        if show:
            cv2.imshow(f'Cropped {i}', crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save crops
        if save_path:
            cv2.imwrite(f'{save_path}cropped_{i}.jpg', crop)

        crops.append(crop)

    return crops
