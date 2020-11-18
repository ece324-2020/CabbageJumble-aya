import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    From Yong Da Li's github. He stole it from someone else. The 2nd-4th parameters are optional.
    """

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def argmax_contour_area(contours):
    """
    Definition:
    max_contour_area(contours)
    Inputs:
    1. contours = list of contours (edges of a coloured region)
    Functionality:
    1. Finds maximum contour area
    Outputs:
    1. max_area_index = index of maximum contour
    """

    max_area = 0
    max_area_index = 0

    for i in range(len(contours)):
        new_area = cv2.contourArea(contours[i])

        if new_area > max_area:
            max_area = new_area
            max_area_index = i

    # Return maximum area's index
    return max_area_index


def argmax_inner_area(contours, hierarchy, max_area_index):
    """
    Definition:
    find_max_inner_area(contours, hierarchy, max_area_index)
    Inputs:
    1. contours = list of contours
    2. hierarchy = tree structure of inner contours
    3. max_area_index = which contour is the parent node
    Functionality:
    >>Runs through inner contours
    1. Find maximum contour
    <<
    Outputs:
    1. max_child_index = child contour with largest area
    Notes:
    hierarchy = [ [[Next, Previous, First_Child, Parent] ] ]
    We need to unpackage the hierarchy
    """

    # Select first inner area from largest area contour
    next_child = hierarchy[0][max_area_index][2]

    # Find max inner contour area
    max_area = 0
    max_child_index = 0

    while (next_child != -1):

        new_area = cv2.contourArea(contours[next_child])

        if new_area > max_area:
            max_area = new_area
            max_child_index = next_child

        next_child = hierarchy[0][next_child][0]  # Select next child

    return max_child_index


def order_inner_area(contours, hierarchy, max_area_index):
    # Select first inner area from largest area contour
    next_child = hierarchy[0][max_area_index][2]

    # Construct List of Areas

    area = []
    index = []

    while (next_child != -1):

        area.append(cv2.contourArea(contours[next_child]))
        index.append(next_child)

        next_child = hierarchy[0][next_child][0]  # Select next child

    # Convert to numpy
    area = np.array(area)
    index = np.array(index)

    # Argsort
    argsort_area = np.argsort(area)[::-1]
    index = index[argsort_area]

    return index, area[argsort_area]


def remove_small(index, area, threshold=2000):
    return index[area > threshold]


def load_images_from_folder(folder):
    # From https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Get list of first-ten-images
images = load_images_from_folder('David_images/David - first-ten-images/raw')
#print(first-ten-images)

count = 295
count1 = 108
count2 = 25
for i, img_original in enumerate(images):
    # Resize so it fits on screen
    img = ResizeWithAspectRatio(img_original, width=600)

    # Create grey image
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_grey = cv2.medianBlur(im_grey, 5)
    im_thresh = cv2.adaptiveThreshold(im_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Contours and hierarchy
    contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_ind = argmax_contour_area(contours)
    child = argmax_inner_area(contours, hierarchy, max_ind)


    # Get sorted index/ areas
    index, area = order_inner_area(contours, hierarchy, max_ind)
    contours = np.array(contours)
    large_children = remove_small(index, area, 2000)

    # Circle LARGE CHILDREN
    all_x = []
    all_y = []
    all_r = []
    for child in large_children:
        # From https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        (x, y), r = cv2.minEnclosingCircle(contours[child])
        scale = img_original.shape[0] / img.shape[0]
        all_x.append(int(x*scale))
        all_y.append(int(y*scale))
        all_r.append(int(r*scale))
        centre = (int(x), int(y))
        r = int(r)
        cv2.circle(img, centre, r, (0, 0, 255), 3)

        # Draw bounding rect
        x, y, w, h = cv2.boundingRect(contours[child])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow(f'Image {i}', img)  # All circled
    key = cv2.waitKey(0)
    if key == ord('1'):
        cv2.imwrite(f'Final_images/{count}.jpg', img_original)
        with open(f'Final_labels/{count}.txt', 'a') as f: 	
            for i in range(len(all_x)):
	            f.write(f'{all_x[i]}\t{all_y[i]}\t{all_r[i]}\n')
        print(count)
        count+=1

    elif key == ord('2'):
        cv2.imwrite(f'meh_images/{count1}.jpg', img_original)
        with open(f'meh_images_labels/{count1}.txt', 'a') as f: 	
	        for i in range(len(all_x)):
	            f.write(f'{all_x[i]}\t{all_y[i]}\t{all_r[i]}\n')
        #print(count1)
        count1+=1
    else:
        cv2.imwrite(f'bad_images/{count2}.jpg', img_original)
        count2+=1

    cv2.destroyAllWindows()

