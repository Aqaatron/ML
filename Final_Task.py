
# coding: utf-8
# # Обработка данных
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


features = pd.read_csv('desktop/features.csv', index_col='match_id')
data = features.loc[:,'start_time':'dire_first_ward_time']
data = data.fillna(0)
data.count()
data = data.fillna(0)
data.count()
target = features.loc[:, 'radiant_win']


# # Часть 1 Градиентный бустинг
kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
n_estimators_array = [10, 20, 30, 40, 50]

scores = []
for k in n_estimators_array:
    mdl = GradientBoostingClassifier(n_estimators=k, verbose=False, random_state=241)
    score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))
    scores.append(score)
scores


mdl = GradientBoostingClassifier(n_estimators=50, verbose=False, random_state=241)
score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))

scores

features_test = pd.read_csv('desktop/features_test.csv', index_col='match_id')
features_test.count()
features_test= features_test.fillna(0)
features_test.count()

mdl.fit(data, target)
pred = mdl.predict_proba(features_test)[:, 1]

pred

# # Часть 2 Логистическая регрессия

data = features.loc[:,'start_time':'dire_first_ward_time']
data = data.fillna(0)
scaler = StandardScaler()
scaler.fit(data)
data_scaler = scaler.transform(data)

data

from sklearn.linear_model import LogisticRegression
C_arr = [10 ** x for x in range(-3, 3, 1)]
scores = []
for C in C_arr:
    mdl = LogisticRegression(penalty='l2', random_state=241, C=C)
    score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))
    scores.append(score)

scores


mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
score = np.mean(cross_val_score(estimator=mdl, X=data_scaler, y=target, cv=kf, scoring='roc_auc'))

score

drop_array = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero','d1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

data_without_categ = data
for k in range(0,len(drop_array)):
    data_without_categ = data_without_categ.drop(drop_array[k],1)
data_without_categ


kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
score = np.mean(cross_val_score(estimator=mdl, X=data_without_categ, y=target, cv=kf, scoring='roc_auc'))

score

scaler.fit(data_without_categ)
data_without_categ_scaler = scaler.transform(data_without_categ)

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
score = np.mean(cross_val_score(estimator=mdl, X=data_without_categ_scaler, y=target, cv=kf, scoring='roc_auc'))

score

heroes = pd.Series()

for h in ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']:
    heroes = heroes.append(data[h])

X_pick = np.zeros((data.shape[0], heroes.max()))

for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick

X=np.hstack([data_without_categ, X_pick])

X

mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
score = np.mean(cross_val_score(estimator=mdl, X=X, y=target, cv=kf, scoring='roc_auc'))

score


scaler.fit(X)
X_scaler = scaler.transform(X)

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
score = np.mean(cross_val_score(estimator=mdl, X=X_scaler, y=target, cv=kf, scoring='roc_auc'))

score

features_test = pd.read_csv('desktop/features_test.csv', index_col='match_id')
features_test= features_test.fillna(0)
features_test_without_categ = features_test
for k in range(0,len(drop_array)):
    features_test_without_categ = features_test_without_categ.drop(drop_array[k],1)
features_test_without_categ


X_pick_test = np.zeros((features_test.shape[0], heroes.max()))

for i, match_id in enumerate(features_test.index):
    for p in range(5):
        X_pick_test[i, features_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick_test


X_test=np.hstack([features_test_without_categ, X_pick_test])
scaler.fit(X_test)
X_test_scaler = scaler.transform(X_test)

mdl.fit(X_scaler, target)
pred = mdl.predict_proba(X_test_scaler)[:,1]

pred

results = pd.DataFrame(
    index=features_test.index,
    data=pred,
    columns=['radiant_win']
)


results.sort_values(by='radiant_win')

results.to_csv('predictions.csv')

