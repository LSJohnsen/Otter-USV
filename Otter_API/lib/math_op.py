
import numpy as np



# Remove elements to change 6 DOF matrices to 3 DOF for Mass and
def reduceDOF(matrix): 
    return np.delete(np.delete(matrix, (2, 3, 4), 1), (2, 3, 4), 0)


