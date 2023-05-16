import numpy as np


def evaluate(net, X, Y):
    Z = net.predict(X, batch_size=10000).flatten()
    Zbin = (Z > 0.5)
    diff = Y - Z
    mse = np.mean(diff*diff)
    n = len(Z)
    n0 = np.sum(Y == 0)
    n1 = np.sum(Y == 1)
    acc = np.sum(Zbin == Y) / n
    tpr = np.sum(Zbin[Y == 1]) / n1
    tnr = np.sum(Zbin[Y == 0] == 0) / n0
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse)
    return acc


def evaluate_mult_pairs(net, cipher, x, y, n_pairs=1):
    # get prediction
    x = x.reshape(-1, 2 * cipher.get_block_size())
    z_atomic = net.predict(x, batch_size=10000).flatten()
    z_atomic = z_atomic.reshape(-1, n_pairs)
    # combine scores under independence assumption
    prod_enc = z_atomic.prod(axis=1)
    prod_rnd = (1 - z_atomic).prod(axis=1)
    # avoid division by zero
    z = 1 / (1 + np.divide(prod_rnd, prod_enc, out=np.zeros_like(prod_enc), where=(prod_enc != 0)))
    # decide score based on the number of zeros in numerator and denominator
    z[prod_enc == 0] = np.sum(z_atomic[prod_enc == 0] == 0, axis=1) < np.sum(z_atomic[prod_enc == 0] == 1, axis=1)

    # evaluate accuracy, tpr, tnr and mse
    z_bin = (z > 0.5)
    diff = y - z
    mse = np.mean(diff * diff)
    n = len(z)
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    acc = np.sum(z_bin == y) / n
    tpr = np.sum(z_bin[y == 1]) / n1
    tnr = np.sum(z_bin[y == 0] == 0) / n0
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse)
    return acc
