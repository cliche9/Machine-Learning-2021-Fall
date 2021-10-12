import numpy as np

def map_feature(feat1, feat2):
    degree = 6
    out = np.ones(feat1.size)

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.column_stack( (out, (feat1 ** (i - j)) * (feat2 ** j)) )

    return out