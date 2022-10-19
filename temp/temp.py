import numpy
import numpy as np

x = np.array([[[1, 2], [2, 3]],[[2, 4], [1, 3]],[[4, 5], [6, 1]]])
print(np.sum(x, axis=2))
