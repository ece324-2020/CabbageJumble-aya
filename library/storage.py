import pickle


def save(var, file):
    """
    Save {var} to {file}.
    :param var: variable value
    :param file: file name
    :return: None
    """
    with open(file, 'wb') as f:
        pickle.dump(var, f)


def load(file):
    """
    Load {file} to {var}.
    :param file: file name
    :return: variable value
    """
    with open(file, 'rb') as f:
        var = pickle.load(f)

    return var