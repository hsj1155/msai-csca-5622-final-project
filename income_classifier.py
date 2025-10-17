# import libraries
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import clone 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# fetch dataset 
adult = fetch_ucirepo(id=2)


# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 


# use 'capital-gain' and 'capital-loss' features to create 'investment-portfolio' feature, then drop them along with 'education-num' and 'fnlwgt'
X['investment-portfolio-gain'] = np.where(X['capital-gain'] == 0, 0, 1)
X['investment-portfolio-loss'] = np.where(X['capital-loss'] == 0, 0, 1)

X['investment-portfolio'] = X['investment-portfolio-gain'] + X['investment-portfolio-loss']
X = X.drop(labels=['investment-portfolio-gain', 'investment-portfolio-loss'], axis=1)
print(X['investment-portfolio'].value_counts())

X = X.drop(labels=['education-num', 'fnlwgt', 'capital-gain', 'capital-loss'], axis=1)

# drop na rows and match y to X
X = X.dropna()
y = y.loc[X.index]


# preserve DataFrame version of y and flatten y into {-1, 1}
y_stable = y
y_int = y

y_int = np.where(y_int == "<=50K", 1, y_int)
y_int = np.where(y_int == "<=50K.", 1, y_int)
y_int = np.where(y_int == 1, int(1), int(-1))

y_new = []

for i in range(len(y)):
    if y_int[i] == [1]:
        y_new.append(1)
    else:
        y_new.append(-1)

y_int = y_new


#turn categorical features into boolean values and drop categorical features
X_sample = pd.concat([X, pd.get_dummies(X['workclass'], prefix='workclass'), pd.get_dummies(X['education'], prefix='education'), pd.get_dummies(X['marital-status'], prefix='marital-status'), pd.get_dummies(X['occupation'], prefix='occupation'), pd.get_dummies(X['relationship'], prefix='relationship'), pd.get_dummies(X['race'], prefix='race'), pd.get_dummies(X['sex'], prefix='sex'),pd.get_dummies(X['native-country'], prefix='native-country')], axis=1)
X_sample = X_sample.drop(labels=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'], axis=1)


#collect scores for DecisionTreeClassifier for different max depths
max_depth_scores = pd.DataFrame({'index': [], 'score': []})

for i in range(20):
    dt = DecisionTreeClassifier(random_state=0, max_depth = i + 1)

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_int, test_size=0.2)
    dt.fit(X_train, y_train)
    print(f'score at max depth {i + 1}: {dt.score(X_test, y_test)}')

    new_row = pd.DataFrame({'index': [i+1], 'score': [1 - dt.score(X_test, y_test)]})
    max_depth_scores = pd.concat([max_depth_scores, new_row])


#create adaboost and collect scores for different numbers of classifiers
clf = AdaBoost(n_learners=150, base=DecisionTreeClassifier(max_depth=1)).fit(X_train, y_train)
scores = clf.staged_score(X_test, y_test)


#plot adaboost staged scores
plt.plot(scores)
plt.title('Staged Score')
plt.xlabel('Stumps')
plt.ylabel('Classification error')
plt.xticks([30, 60, 90, 120, 150])
plt.show()


#create random forest and determine score
y=y.replace('<=50K', 1)
y=y.replace('<=50K.', 1)
y=y.replace('>50K', -1)
y=y.replace('>50K.', -1)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y, test_size=0.2)
rf = RandomForest(X_train, y_train, 100)
print(rf.score(X_test, y_test))
