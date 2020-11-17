import numpy as np


for i in range(1, 576+1):
    path = f'../../data/final_labels/{i}.txt'
    a = np.loadtxt(path, dtype=int, delimiter='\t', ndmin=2)
    b = np.zeros((len(a), 5))

    if a.shape[1] == 3:
        b[:, :3] = a
    else:
        b = a

    np.savetxt(path, b, fmt='%i', delimiter='\t')

    print(b)
