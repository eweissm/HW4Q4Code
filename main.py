import numpy as np

x1 = 1  # initial driving condition

x2 = 1  # initial state conditions
x3 = 1

X = np.array([x2, x3, x1])

k = 0  # initialize counter for solver
e = 10 ** -3  # minimum error for slope

sk = np.array([X[0], X[1]])  # state variables
dk = np.array([X[2]])  # decision variable

dfdx1 = np.array([2 * X[2]])  # dfdd (partial)
dfds = np.array([2 * X[0], 2 * X[1]])  # dfds (partial)
dhds = np.array([[2 / 5 * X[0], 2 / 25 * X[1]],
                 [1, -1]])

dhdd = np.array([[1 / 2 * X[2]], [1]])
dfdd = dfdx1 - np.matmul(np.matmul(dfds, np.linalg.inv(dhds)), dhdd)  # non-linear dfdd

