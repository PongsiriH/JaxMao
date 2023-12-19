# https://arxiv.org/pdf/2009.14193.pdf

import numpy as np

# Algorithm 1:
def naive_prediction_sets(alpha, sorted_scores, associated_classes, rand=False):
    L = 1
    while np.sum(sorted_scores[:L]) < 1 - alpha:
        L += 1
        if L > len(sorted_scores):
            break

    if rand and L < len(sorted_scores):
        U = np.random.uniform(0, 1)
        V = (np.sum(sorted_scores[:L]) - (1 - alpha)) / sorted_scores[L-1]
        if U <= V:
            L -= 1

    return associated_classes[:L]

# Algorithm 2
def raps_conformal_calibration(alpha, s, I, y, k_reg, lam, rand=False):
    # s: (n, K) = (num_sample, num_classes)
    # I: associated permutation of indexes
    # y: ground truth
    # k_reg, lam are regularization terms.
    n, K = s.shape
    E = np.zeros(n)

    for i in range(n): # can use vmap?
        L_i = np.where(I[i] == y[i])[0][0]  # Li: index of the true label in the sorted list
        E[i] = np.sum(s[i, :L_i + 1]) + lam * (L_i - k_reg)
        if rand:
            U = np.random.uniform(0, 1)
            E[i] -= U * s[i, L_i]

    print(E, (1 - alpha) * (1 + n), int(np.ceil((1 - alpha) * (1 + n)) - 1))
    index = int(np.ceil((1 - alpha) * (1 + n)))
    tau_hat_ccal = np.sort(E)[index - 1 - 1]
    return tau_hat_ccal

# Example naive_prediction_sets
if __name__ == '__main__':
    alpha = 0.05
    scores = np.array([0.45, 0.02, 0.18, 0.35])
    associated_classes = np.array(['A', 'B', 'C', 'D'])
    
    # sorting from high to low.
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_classes = associated_classes[sorted_indices]
    
    prediction_set = naive_prediction_sets(alpha, sorted_scores, sorted_classes, rand=True)
    print("Prediction Set:", prediction_set)


# Example raps_conformal_calibration
if __name__ == '__main__':
    print('\n\n**raps_conformal_calibration**')
    alpha = 0.05
    s = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    I = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])  
    y = np.array([2, 3]) 
    k_reg = 1
    lam = 0.1

    tau_hat_ccal = raps_conformal_calibration(alpha, s, I, y, k_reg, lam, rand=True)
    print("Generalized quantile:", tau_hat_ccal)