import numpy as np
from collections import Counter

def norm(x, y):
    return sum((x-y)**2)**0.5

def mitigate(labels):
    counts = Counter(labels)
    max_key = max(counts, key=counts.get)

    return max_key


class KNearestNeighbor:

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def compute_distances_two_loops(self, X_test):
        dist_matrix = []
        for x_test in X_test:
            row = []
            for x_train in self.X_train:
                row.append(norm(x_test, x_train))
            dist_matrix.append(np.array(row))

        return np.array(dist_matrix)
    
    def compute_distances_one_loop(self, X_test):
        dist_matrix = []
        for i, x in enumerate(X_test):
            diff = self.X_train - x
            dist_matrix.append(np.sqrt(np.sum(diff ** 2, axis=1)))

        return dist_matrix
    
    def compute_distances_no_loops(self, X_test):
        return np.sqrt(np.sum(X_test**2, axis=1)[:, np.newaxis] + np.sum(self.X_train**2, axis=1)[np.newaxis, :] - 2 * X_test @ self.X_train.T)


    def predict_labels(self, dists, k=1):
        all_labels = []
        for curr_dists in dists:
            order = np.argsort(curr_dists)
            dists_with_ind = list(zip(order.tolist(), curr_dists[order].tolist()))
            cur_labels = []
            for dist_ind in dists_with_ind[:k]:
                cur_labels.append(self.y_train[dist_ind[0]])
            all_labels.append(mitigate(cur_labels))


        return np.array(all_labels)

    def predict(self, X, k=1):
        dists = self.compute_distances_no_loops(X)
        return self.predict_labels(dists, k=k)