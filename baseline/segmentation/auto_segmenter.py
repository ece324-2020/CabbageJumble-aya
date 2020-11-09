import numpy as np
import cv2
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.load_images_from_folder import load_images_from_folder
from library.contours import argmax_contour_area, children_area, arg_large_areas


images = load_images_from_folder('../../data/david/raw')

for i, img_original in enumerate(images):
    # Resize so it fits on screen
    img = ResizeWithAspectRatio(img_original, width=600)

    # Create grey image
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_grey = cv2.medianBlur(im_grey, 5)
    im_thresh = cv2.adaptiveThreshold(im_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Contours and hierarchy
    contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    argmax_area = argmax_contour_area(contours)

    # Get sorted index/ areas
    index, area = children_area(contours, hierarchy, argmax_area)
    contours = np.array(contours)
    large_children = arg_large_areas(index, area, 2000)

    labels = ''

    # Circle LARGE CHILDREN
    for child in large_children:
        # From https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        (x, y), r = cv2.minEnclosingCircle(contours[child])
        centre = (int(x), int(y))
        r = int(r)
        cv2.circle(img, centre, r, (0, 0, 255), 3)

        scale = img_original.shape[0] / img.shape[0]
        labels += f'{round(x * scale)}\t{round(y * scale)}\t{round(r * scale)}\n'

        # Draw bounding rect
        x, y, w, h = cv2.boundingRect(contours[child])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show scaled image with coins circle
    cv2.imshow(f'Image {i}', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save ORIGINAL IMAGE and LABEL
    if key == ord(' '):
        cv2.imwrite(f'../../data/david/good_segment/good_img_{i}.jpg', img_original)
        with open(f'../../data/david/good_segment/good_label_{i}.txt', 'w') as f:
            f.write(labels)
    elif key == ord('k'):
        cv2.imwrite(f'../../data/david/meh_segment/meh_img_{i}.jpg', img_original)
        with open(f'../../data/david/good_segment/good_label_{i}.txt', 'w') as f:
            f.write(labels)
    else:
        cv2.imwrite(f'../../data/david/bad_segment/bad_img_{i}.jpg', img_original)
