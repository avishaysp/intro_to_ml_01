import numpy as np
import matplotlib.pyplot as plt

N = 200000
n = 20


def gen_matrix(size: tuple) -> np.ndarray:
    return np.random.choice([0, 1], size=size, p=[0.5, 0.5])


def get_means(matrix: np.ndarray):
    return np.mean(matrix, axis=1)


# section (a)
mat = gen_matrix((N, n))
means = get_means(mat)


# section (b)
eps = np.linspace(0, 1, 50)
empirical_probabilities = [np.sum(abs(means - 0.5) > e) / N for e in eps]
empirical_probabilities = np.array(empirical_probabilities)


plt.plot(eps, empirical_probabilities, label='Empirical Probability')
plt.xlabel('Epsilon')
plt.ylabel('Empirical Probability')
plt.title('Visualizing the Hoeffding bound')
plt.legend()
plt.show()
