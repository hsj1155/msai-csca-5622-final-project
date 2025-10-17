class RandomForest():
    
    def __init__(self, x, y, sample_sz, n_trees=2, n_features='sqrt', max_depth=10, min_samples_leaf=5):
        np.random.seed(42)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        print(self.n_features, "sha: ",x.shape[1])  
        self.features_set = []
        self.x, self.y, self.sample_sz, self.max_depth, self.min_samples_leaf  = x, y, sample_sz, max_depth, min_samples_leaf
        self.trees = [self.create_tree(i) for i in range(n_trees)]
        
    def create_tree(self,i):
        idxs = self.x.sample(n=self.sample_sz).index
        idxs = np.asarray(idxs)

        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        f_idxs = np.asarray(f_idxs)  

        if i==0:
            self.features_set = np.array(f_idxs, ndmin=2)
        else:
            self.features_set = np.append(self.features_set, np.array(f_idxs,ndmin=2),axis=0)

        clf = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf = self.min_samples_leaf)        
        
        x = pd.DataFrame()
        y = pd.DataFrame()
        
        for i in range(len(f_idxs)):
            x = pd.concat([x, self.x.iloc[:, f_idxs[i]]], axis=1)

        y = pd.concat([y, self.y], axis=1)

        x = x.loc[idxs]
        y = y.loc[idxs]

        clf.fit(x, y)

        return clf
       
    def predict(self, x):

        yhat = np.zeros(x.shape[0], dtype=int)
        yhat_float = np.zeros(x.shape[0])

        for i in range(len(self.trees)):
            x_sub_i = pd.DataFrame()
            for j in range(len(self.features_set[i])):
                x_sub_i = pd.concat([x_sub_i, x.iloc[:, self.features_set[i][j]]], axis=1) 
            
            yhat_sub_i = np.zeros(x.shape[0], dtype=int)            
            yhat_sub_i = self.trees[i].predict(x_sub_i)
            yhat = yhat + yhat_sub_i
        
        for i in (range(len(yhat))):
            if yhat[i] > 0: 
                yhat[i] = 1
            else:
                yhat[i] = -1
            
        return yhat
    
    def score(self, X, y):

        yhat = self.predict(X)

        total_right = 0
        total_wrong = 0
        
        for i in range(len(y)):
            true_val = y.iloc[i, 0]

            if true_val == yhat[i]:
                total_right += 1
            else:
                total_wrong += 1
        
        print(f'total right: {total_right}; total wrong: {total_wrong}')
        
        return(1 - total_right/(total_right+total_wrong))
