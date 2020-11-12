import numpy as np


def save_circle_coord(path, labels, scale: int = 1):
    """
    Saves a array of circle coordinates as a text file.
    :param labels: label
    :param path: path to file where we want to save
    :param scale: scale by which we multiply
    :return:
    """
    # Convert iterable to string
    if isinstance(labels, (list, np.ndarray)):
        # Convert to string
        str_labels = ''
        for label in labels:
            # Add default three
            str_labels += f'{round(label[0] * scale)}\t{round(label[1] * scale)}\t{round(label[2] * scale)}'

            # Add extras
            if len(label) == 4:
                str_labels += f'\t{round(label[3])}'
            elif len(label) == 5:
                str_labels += f'\t{round(label[3])}\t{round(label[4])}'

            # And linebreak
            str_labels += '\n'

        # Make labels into str
        labels = str_labels

    with open(path, 'w') as f:
        f.write(labels)


def load_circle_coord(file_path, order_size: bool = False):
    """
    Load the coordinates of a text file with circle labels.

    :param file_path: str - path to text file. Relative to the file you are currently in.
        e.g. 'data/labels/104.txt'

        The text file should be formatted:
        {x}\t{y}\t{r}(\t{value})(\t{H_or_T})
        where the last two values are optional.
        e.g.
        99      99      20      100     72
    :param order_size: bool - whether to order from smallest to largest.
    :param H_or_T: bool - whether Heads or Tails is present
    :return: np.array 2D -
        e.g. [[x, y, r], [x, y, r], ...]
    """
    with open(file_path, 'r') as f:
        labels = f.read()

    # Break between lines
    labels = labels.strip().split('\n')

    # Get each (x, y, r, value, H/T)
    labels = np.array([list(map(int, label.split('\t'))) for label in labels])

    # OPTIONAL: Sort in order of increasing size
    if order_size:
        index = np.argsort(labels[:, 2], axis=0)[::-1]
        labels = labels[index]

    return labels


def decode_HT(entry):
    """
    Decode whether an entry is Heads or Tails
    :param entry: scalar, iterable - entry to decode
    :return: str - ASCII letter
    """

    # Check if string
    if isinstance(entry, (int, float)):
        HT = chr(int(entry))

    # Else, assume iterable
    else:
        HT = chr(entry[4])

    return HT


def add_value_and_HT(file_path, write_to_file: bool = False, default_value: int = 0, default_HT: str = 'H'):
    """
    Add value and Heads/Tails to label. Default is 0 value and Heads.

        N.B. Store H/T in ASCII: H = 72, T = 84.

    :param file_path: str - path to file
    :param write_to_file: bool - whether to write result to the file
    :param default_value: int - default monetary value (in cents)
    :param default_HT: str - default character for heads/ tails ('H' or 'T')
    :return: np.array of labels
    """

    # Open file path in read mode
    with open(file_path, 'r') as f:
        labels = f.read()

    labels = labels.strip().split('\n')

    # Create list, not necessarily of same size-- all must be integers!
    labels = [list(map(int, label.split('\t'))) for label in labels]

    # Create switch case
    switch = {
        3: lambda label: label + [default_value, ord(default_HT)],
        4: lambda label: label + [ord(default_HT)],
        5: lambda label: label
    }

    # Add to label
    for i, label in enumerate(labels):
        # None if invalid length
        labels[i] = switch.get(len(label), None)(label)

    # If there is a None in the labels, raise exception
    if not all(labels):
        raise Exception('Incorrect length in text file')

    # Create np.array
    labels = np.array(labels)

    # Write to file
    if write_to_file:
        np.savetxt(file_path, labels, fmt='%i', delimiter='\t')

    return labels
