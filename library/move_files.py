import os
import shutil

def move_files(path_from, path_to, number: int = 200):
    """
    Move number of files from path_from to path_to.
    :param path_from: str - path from
    :param path_to: str - path to
    :param number: int - if non-positive, it will run to infinity
    :return:
    """
    files_accum = []

    # Get file names to be moved
    for path, subdirs, files in os.walk(path_from):
        for file in files:                         # name.jpg
            files_accum.append(os.path.join(path, file))
            # Exit when we have 'number' files
            if len(files_accum) == number:
                break

    # Move files - must be in different loop, otherwise walk tries to pick up these images too!
    for current_file in files_accum:
        base_file = os.path.basename(current_file)
        os.replace(current_file, os.path.join(path_to, base_file))

    return files_accum

if __name__ == '__main__':
    path_from = '$ scrap_data'
    path_to = '../data/seg_image'

    # path_from = '../data/seg_image'
    # path_to = '$ scrap_data'

    move_files(path_from, path_to, 2)