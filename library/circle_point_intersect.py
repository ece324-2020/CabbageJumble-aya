import numpy as np
import warnings


def match_points_to_circles(circles: np.ndarray, values: tuple):
    """
    Match circles to points in a single file.

    :param circles: np.array or list - [[x, y, r, (val, HT)], []].
    :param values: tuple - ('img_name.jpg', {(x, y): [value, HT], (x, y): [value, HT]})
        - Obtain via list(dict.items())[i]

    :return: new_circles
    """

    # Check if circles is an np.array
    if not isinstance(circles, np.ndarray):
        warnings.warn('circles in not an np.ndarray. I fixed it for you. You\'re welcome.')
        circles = np.array(circles)

    # Extract img_name and points
    if isinstance(values, dict):
        values = next(iter(values.items()))
    img_name, points = values

    # Initialize good storage
    if circles.shape[1] == 3:
        new_circles = np.pad(circles, ((0, 0), (0, 2)))
    elif circles.shape[1] == 5:
        warnings.warn(f'Overwriting {img_name}\'s existing values. It is the correct size!')
    else:
        warnings.warn('circles is wrong shape (must be 3 or 5)')
        new_circles = np.zeros((len(circles), 5))

    # Check for collisions by recording which are filled
    done = np.zeros(len(circles), dtype=bool)

    for i, point in enumerate(points):
        # Find the circle index (k) that the point is closest to
        k = find_matching_circle(circles, point)

        # Check if this is valid
        if k is None:
            warnings.warn(f'Missing circle in {img_name} for point {point}. Could not find circle!')
        elif done[k]:
            warnings.warn(f'Collision in {img_name} for point {point} in position {k}. Prevented overwrite.')
        else:
            new_circles[k, -2:] = points[point]

            # Mark as finished
            done[k] = True

    return new_circles


def find_matching_circle(circles: np.ndarray, point: np.ndarray):
    """
    Find which circles a point intersects.


    :param circles: np.array - [[x, y, r, (val, HT)], []]
    :param point: tuple, list, or np.array - [x, y]

    :return: bool
    """

    # Ensure circles is an np.array
    if not isinstance(circles, np.ndarray):
        warnings.warn('circles in not an np.ndarray. I fixed it for you. You\'re welcome.')
        circles = np.array(circles)

    # Calculate distances
    dist = np.linalg.norm(np.array(point) - circles[:, :2], axis=1)

    # Find minimum distance
    argmin = np.argmin(dist)

    # Check if minimum distance is small enough
    if dist[argmin] <= circles[argmin, 2]:
        return argmin
    else:
        return None


if __name__ == '__main__':
    circles = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    squares = {'image_name.jpg': {(1, 2): [100, 72], (4, 5): [200, 84], (7, 8): [300, 99]}}
    new_circles = match_points_to_circles(circles, squares)
