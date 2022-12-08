import numpy as np

x1 = 1  # initial driving condition

x2 = 1  # initial state conditions
x3 = 1

X = np.array([x2, x3, x1])

k = 0  # initialize counter for solver
e = 10 ** -3  # minimum error for slope

maxLineseachSteps = 50
maxRalphsonSteps = 30

sk = np.array([X[0], X[1]])  # state variables
dk = np.array([X[2]])  # decision variable

dfdx1 = np.array([2 * X[2]])  # dfdd (partial)
dfds = np.array([2 * X[0], 2 * X[1]])  # dfds (partial)
dhds = np.array([[2 / 5 * X[0], 2 / 25 * X[1]],
                 [1, -1]])

dhdd = np.array([[1 / 2 * X[2]], [1]])
dfdd = dfdx1 - np.matmul(np.matmul(dfds, np.linalg.inv(dhds)), dhdd)  # non-linear dfdd

def f(X):
    x1 = X[2]
    x2 = X[0]
    x3 = X[1]
    return x1 ** 2 + x2 ** 2 + x3 ** 2

def constraints(X):  # constraint
    x1 = X[2]
    x2 = X[0]
    x3 = X[1]
    h1 = x1 ** 2 / 4 + x2 ** 2 / 5 + x3 ** 2 / 25 - 1
    h2 = x1 + x2 - x3
    return np.array([[h1], [h2]])

def linesearch(dfdd, sk, dk, dhds_line, dhdd_line):  # inexact line search
    alpha = 1

    if dfdd < 0:
        alpha = -0.1

    #line search parameters
    b = 0.4
    t = 0.2

    i = 0

    dhds_inv = np.linalg.inv(dhds_line)  # calculates inverse of dhds for function evaluation

    skstep_matrix = np.transpose(np.matmul(np.matmul(dhds_inv, dhdd_line), dfdd))
    sk1 = sk + alpha * skstep_matrix  # state step
    dk1 = dk - alpha * dfdd  # driving step
    X_step = np.concatenate((sk1, dk1), axis=None)
    funct = f(X_step)
    phi = f(np.concatenate((sk, dk), axis=None)) - alpha * t * (dfdd)**2  # phi with step
    while funct > phi and i < maxLineseachSteps:
        alpha = b * alpha #shrink step size
        sk1 = sk + alpha * skstep_matrix
        dk1 = dk - alpha * dfdd
        X_step = np.concatenate((sk1, dk1), axis=None)
        funct = f(X_step)
        phi = f(np.concatenate((sk, dk), axis=None)) - alpha * t * (dfdd)**2 #recalc for new alpha value
        i += 1

    return alpha, skstep_matrix

# Newton ralphson
def solvefunc(decision, state_approx):
    c = 0
    x1 = decision
    x2 = state_approx[0]
    x3 = state_approx[1]
    skactual = np.array([x2, x3])
    Xpart = np.concatenate((state_approx, decision), axis=None)
    h = constraints(Xpart)
    while np.linalg.norm(h) > 10 ** -3 and c < maxRalphsonSteps:
        sktran = np.reshape(skactual, (2,1)) - np.matmul(
            np.linalg.inv(np.array([[2 / 5 * Xpart[0], 2 / 25 * Xpart[1]], [1, -1]])), constraints(Xpart))
        x2 = sktran[0]
        x3 = sktran[1]
        skactual = np.concatenate((x2, x3), axis=None)
        Xpart = np.concatenate((x2, x3, x1), axis=None)
        h = constraints(Xpart)
        c += 1

    return skactual