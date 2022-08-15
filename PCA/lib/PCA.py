import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None
        self.type = ["gda", "sga"]

    def fit(self, X, eta=0.01, n_iters=1e4, type="gda"):
        """获得数据集X的前n_components (k) 个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        assert type in self.type, "unknown pca type, choose gda or sga"

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X @ w) ** 2) / len(X)

        def df(w, X):
            return X.T @ (X @ w) * 2. / len(X)

        def dJ_sgd(w, X_i):
            return (X_i.T * (X_i @ w)) * 2 / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component_by_gda(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        def first_component_by_sgd(X, initial_w, n_iters=5, t0=5, t1=50):
            assert X.shape[1] == initial_w.shape[0], \
                "the number of features of X must be equal to the size of w"
            assert n_iters >= 1

            def learning_rate(t):
                return t0 / (t + t1)

            w = direction(initial_w)
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(len(X))
                X_new = X[indexes]
                for i in range(len(X)):
                    gradient = dJ_sgd(w, X_new[i])
                    w = w + learning_rate(cur_iter * len(X) + i) * gradient
                    w = direction(w)
            return w

        def first_component(pca_method, X_pca, initial_w, eta=0.01, n_iters=1e4):
            if pca_method == "gda":
                return first_component_by_gda(X_pca, initial_w, eta, n_iters)
            if pca_method == "sga":
                return first_component_by_sgd(X_pca, initial_w, n_iters)


        # 计算前k个主成分
        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component("gda", X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
