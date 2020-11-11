import cv2
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.store_circle import load_circle_coord


for i in range(410, 471):
    label_path = f'../data/david-labelled/{i}.txt'
    img_path = f'../data/david-labelled/{i}.jpeg'

    # Open labels
    labels = load_circle_coord(label_path)

    # Open image and create mask
    img = cv2.imread(img_path)

    # Draw white circles on mask
    for j, label in enumerate(labels):
        x, y, r = label

        # Draw white circle
        cv2.circle(img, (x, y), r, (255, 0, 0), 3)


    # Resize image
    img = ResizeWithAspectRatio(img, width=600)

    cv2.imshow(f'Image {i}', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 466, 469