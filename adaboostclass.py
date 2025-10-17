class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=3), random_state=1234):
        np.random.seed(42)
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):

        w = np.ones(len(y_train), dtype = np.longdouble)
        w /= np.sum(w)
        
        for k in range(self.n_learners):

            h = clone(self.base)
            h.fit(X_train, y_train, sample_weight=w)
            
            yhat = h.predict(X_train)

            errk = self.error_rate(y_train, yhat, w)
            self.alpha[k] = 0.5 * np.log((1-errk)/errk)
            
            for i in range(len(y_train)):
                identity = 1
                if y_train[i] == yhat[i]:
                    identity = 0
                w[i] = w[i] * np.exp(-1 * self.alpha[k] * yhat[i] * y_train[i])
                
            w /= np.sum(w)                
            self.learners.append(h)
            
        self.base = h
        return self  
            
    def error_rate(self, y_true, y_pred, weights):

        err_num = 0
        err_den = 0

        for i in range(len(y_true)):
            identity = 1
            if y_true[i] == y_pred[i]:
                identity = 0

            err_num += weights[i] * identity
            err_den += weights[i]

        errk = err_num/err_den
        return errk
        
        
    def predict(self, X):

        yhat = np.zeros(X.shape[0], dtype=int)
        yhat_float = np.zeros(X.shape[0])
        
        for i in range(len(self.learners)):
            yhat_float = yhat_float + self.learners[i].predict(X) * self.alpha[i]

        for i in range(len(yhat)):
            if yhat_float[i] >= 0:
                yhat[i] = np.int64(1)
            else:
                yhat[i] = np.int64(-1)
                        
        return yhat
        
    
    def score(self, X, y):
        yhat = self.predict(X)
        
        right_count = 0
        wrong_count = 0

        for i in range(len(y)):
            if y[i] == yhat[i]:
                right_count += 1
            else:
                wrong_count += 1

        #print(right_count, wrong_count)
        return (right_count / (right_count + wrong_count))
        
    
    def staged_score(self, X, y):
        scores = []
        
        yhat = np.zeros(X.shape[0], dtype=int)
        yhat_float = np.zeros(X.shape[0])

        for i in range(len(self.learners)):
            yhat_float = yhat_float + self.learners[i].predict(X) * self.alpha[i]

            right_count = 0
            wrong_count = 0

            for j in range(len(yhat)):
                if yhat_float[j] >= 0:
                    yhat[j] = np.int64(1)
                else:
                    yhat[j] = np.int64(-1)
                if y[j] == yhat[j]:
                    right_count += 1
                else:
                    wrong_count += 1
                
            scores.append(1 - (right_count/(right_count + wrong_count)))

        return scores
