import numpy as np


def dataLoader(X_train, y_train, batch_size=16):
    idx = np.arange(X_train.shape[0])
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]
    for i in range(0, X_train.shape[0], batch_size):
        start, end = i, min(X_train.shape[0], i+batch_size)
        yield X_train[start:end], y_train[start:end]


class RidgeRegression:
    def __init__(self):
        pass

    def fit(self, X_train, y_train, LAMBDA):
        W = np.linalg.inv(X_train.T.dot(X_train) + LAMBDA *
                          np.eye(X_train.shape[1])).dot(X_train.T.dot(y_train))
        return W

    def dataLoader(X_train, y_train, batch_size=16):
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]
        for i in range(0, X_train.shape[0], batch_size):
            start, end = i, min(X_train.shape[0], i+batch_size)
            yield X_train[start:end], y_train[start:end]

    def fit_gradient(self, X_train, y_train, LAMBDA, batch_size, lr, epoch=10):
        W = np.random.randn(X_train.shape[1])
        last_loss = 10e+8
        for _ in range(epoch):
            for x, y in dataLoader(X_train, y_train, batch_size):
                W -= lr * (X_train.T.dot(X_train.dot(W)-y_train) + LAMBDA * W)
            new_loss = self.computeRSS(self.predict(W, X_train), y_train)
            if np.abs(new_loss-last_loss) <= 1e-5:
                break
            last_loss = new_loss
            if _ % 100 == 0:
                print(last_loss)
        return W

    def predict(self, W, X_new):
        return np.array(X_new).dot(W)

    def computeRSS(self, Y_new, Y_pred):
        return np.mean((Y_new-Y_pred)**2)

    def getTheBestLAMBDA(self, X_train, y_train):

        def crossValidation(num_folds, LAMBDA):
            row_ids = np.arange(X_train.shape[0])
            valid_ids = np.split(
                row_ids[:len(row_ids)-len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1],
                                      row_ids[len(row_ids)-len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]]
                         for i in range(num_folds)]
            avg_RSS = 0
            for i in range(num_folds):
                W = self.fit(X_train[train_ids[i]],
                             y_train[train_ids[i]], LAMBDA)
                y_pred = self.predict(W, X_train[valid_ids[i]])
                avg_RSS += self.computeRSS(y_train[valid_ids[i]], y_pred)
                return avg_RSS / num_folds

        def rangeScan(best_LAMBDA, min_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                avg_RSS = crossValidation(num_folds=2, LAMBDA=current_LAMBDA)
                if avg_RSS < min_RSS:
                    best_LAMBDA = current_LAMBDA
                    min_RSS = avg_RSS
            return best_LAMBDA, min_RSS

        best_LAMBDA, min_RSS = rangeScan(
            best_LAMBDA=0, min_RSS=1000**2, LAMBDA_values=range(50))
        LAMBDA_values = np.arange(
            max(0, (best_LAMBDA-1)*1000, (best_LAMBDA+1)*1000, 1))*1.0/1000
        best_LAMBDA, min_RSS = rangeScan(
            best_LAMBDA=best_LAMBDA, min_RSS=min_RSS, LAMBDA_values=LAMBDA_values)

        return best_LAMBDA
