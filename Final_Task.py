
# coding: utf-8

# # Обработка данных
# 

# In[67]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[107]:


features = pd.read_csv('desktop/features.csv', index_col='match_id')
data = features.loc[:,'start_time':'dire_first_ward_time']
data = data.fillna(0)


# In[108]:


data.count()


# In[109]:


data = data.fillna(0)


# In[110]:


data.count()


# In[23]:


target = features.loc[:, 'radiant_win']


# # Часть 1 Градиентный бустинг

# In[27]:


kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)


# In[43]:


n_estimators_array = [10, 20, 30, 40, 50]


# In[44]:


scores = []
for k in n_estimators_array:
    mdl = GradientBoostingClassifier(n_estimators=k, verbose=False, random_state=241)
    mdl.fit(data, target)
    score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))
    scores.append(score)


# In[45]:


scores


# In[47]:


scores = []
mdl = GradientBoostingClassifier(n_estimators=50, verbose=False, random_state=241)
mdl.fit(data, target)
score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))
scores.append(score)    


# In[48]:


scores


# In[224]:


features_test = pd.read_csv('desktop/features_test.csv', index_col='match_id')
features_test= features_test.fillna(0)


# In[50]:


features_test.head()


# In[54]:


features_test= features_test.fillna(0)


# In[55]:


features_test.count()


# In[56]:


pred = mdl.predict_proba(features_test)[:, 1]


# In[63]:


pred


# In[60]:


import time
import datetime

start_time = datetime.datetime.now()

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = GradientBoostingClassifier(n_estimators=30, verbose=False, random_state=241)
mdl.fit(data, target)
score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))

print ('Time elapsed:', datetime.datetime.now() - start_time)


# In[64]:


score


# In[79]:


len(data)


# # Часть 2 Логистическая регрессия

# In[242]:


data = features.loc[:,'start_time':'dire_first_ward_time']
data = data.fillna(0)
scaler = StandardScaler()
scaler.fit(data)
data_scaler = scaler.transform(data)


# In[243]:


data


# In[244]:


import time
import datetime
from sklearn.linear_model import LogisticRegression
C_arr = [10 ** x for x in range(-3, 3, 1)]
start_time = datetime.datetime.now()
scores = []
kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
for C in C_arr:
    mdl = LogisticRegression(penalty='l2', random_state=241, C=C)
    mdl.fit(data, target)
    score = np.mean(cross_val_score(estimator=mdl, X=data, y=target, cv=kf, scoring='roc_auc'))
    scores.append(score)
print ('Time elapsed:', datetime.datetime.now() - start_time)


# In[245]:


scores


# In[246]:


start_time = datetime.datetime.now()
score = []
kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
mdl.fit(data, target)
score = np.mean(cross_val_score(estimator=mdl, X=data_scaler, y=target, cv=kf, scoring='roc_auc'))

print ('Time elapsed:', datetime.datetime.now() - start_time)


# In[247]:


score


# In[248]:


drop_array = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero','d1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']


# In[249]:


data_without_categ = data
for k in range(0,len(drop_array)):
    data_without_categ = data_without_categ.drop(drop_array[k],1)
data_without_categ


# In[250]:


start_time = datetime.datetime.now()

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
mdl.fit(data, target)
score = np.mean(cross_val_score(estimator=mdl, X=data_without_categ, y=target, cv=kf, scoring='roc_auc'))

print ('Time elapsed:', datetime.datetime.now() - start_time)
score


# In[251]:


scaler.fit(data_without_categ)
data_without_categ_scaler = scaler.transform(data_without_categ)


# In[252]:


start_time = datetime.datetime.now()

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
mdl.fit(data, target)
score = np.mean(cross_val_score(estimator=mdl, X=data_without_categ_scaler, y=target, cv=kf, scoring='roc_auc'))

print ('Time elapsed:', datetime.datetime.now() - start_time)
score


# In[253]:


data


# In[254]:


heroes = pd.Series()

for h in ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']:
    heroes = heroes.append(data[h])


# In[255]:


len(heroes.unique())


# In[256]:


X_pick = np.zeros((data.shape[0], heroes.max()))

for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1


# In[257]:


X_pick


# In[258]:


X=np.hstack([data_without_categ, X_pick])


# In[259]:


X


# In[260]:


start_time = datetime.datetime.now()

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
mdl.fit(data, target)
score = np.mean(cross_val_score(estimator=mdl, X=X, y=target, cv=kf, scoring='roc_auc'))

print ('Time elapsed:', datetime.datetime.now() - start_time)
score


# In[261]:


scaler.fit(X)
X_scaler = scaler.transform(X)


# In[288]:


start_time = datetime.datetime.now()

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=241)
mdl = LogisticRegression(penalty='l2', random_state=241, C=0.001)
score = np.mean(cross_val_score(estimator=mdl, X=X_scaler, y=target, cv=kf, scoring='roc_auc'))

print ('Time elapsed:', datetime.datetime.now() - start_time)
score


# In[289]:


features_test = pd.read_csv('desktop/features_test.csv', index_col='match_id')
features_test= features_test.fillna(0)
features_test_without_categ = features_test
for k in range(0,len(drop_array)):
    features_test_without_categ = features_test_without_categ.drop(drop_array[k],1)
features_test_without_categ


# In[290]:


X_pick_test = np.zeros((features_test.shape[0], heroes.max()))

for i, match_id in enumerate(features_test.index):
    for p in range(5):
        X_pick_test[i, features_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick_test


# In[291]:


X_test=np.hstack([features_test_without_categ, X_pick_test])
scaler.fit(X_test)
X_test_scaler = scaler.transform(X_test)
X_test_scaler.shape


# In[292]:


X_scaler.shape


# In[294]:


mdl.fit(X_scaler, target)
pred = mdl.predict_proba(X_test_scaler)[:,1]


# In[295]:


pred


# In[297]:


results = pd.DataFrame(
    index=features_test.index,
    data=pred,
    columns=['radiant_win']
)


# In[301]:


results.sort_values(by='radiant_win')


# In[302]:


results.to_csv('predictions.csv')

