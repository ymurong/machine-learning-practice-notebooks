import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):

    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distances = sqrt(np.sum((X_train - x)**2, axis=1))
    nearest = np.argsort(distances)

    topK_y = y_train[nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]