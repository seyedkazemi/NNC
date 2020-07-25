#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


# THIS PART FOR CHECK RESIDENT 1 tree Decision:

for i in range(29):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(26):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==25:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

dt1 = tree.DecisionTreeClassifier(criterion='gini')  

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
dt1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", dt1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = dt1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")




# THIS PART FOR CHECK RESIDENT 2 tree Decision:


for i in range(29):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(26):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==25:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

dt2 = tree.DecisionTreeClassifier(criterion='gini')  
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
dt2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", dt2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = dt2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")





# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 Gaussian NB:

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

gn1 = GaussianNB() 

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
gn1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", gn1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = gn1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")




# THIS PART FOR CHECK RESIDENT 2 GaussianNB:



for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
      data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

gn2 = GaussianNB() 
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
gn2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", gn2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = gn2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")



# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 KNN:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

knn1= KNeighborsClassifier(n_neighbors = 5)

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
knn1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", knn1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = knn1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")

# THIS PART FOR CHECK RESIDENT 2 KNN:


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

knn2= KNeighborsClassifier(n_neighbors = 5)
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
knn2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", knn2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = knn2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")

aa = statistics.mean(precision)
bb = statistics.mean(recall)
cc = statistics.mean(fscore)

print('precision: ',aa)
print('recall: ',bb)
print('fscore: ',cc)

print("---+++---+++End+++---+++---")


# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 LightGBM:

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

lg1 = LGBMClassifier()

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
lg1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", lg1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = lg1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")




# THIS PART FOR CHECK RESIDENT 2 LightGBM:



for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

lg2 = LGBMClassifier()
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
lg2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", lg2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = lg2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")


# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 SVM:

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

sv1 = svm.SVC(gamma = 0.001, C=100)

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
sv1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", sv1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = sv1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")



# THIS PART FOR CHECK RESIDENT 2 SVM:



for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

sv2 = svm.SVC(gamma = 0.001, C=100)
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
sv2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", sv2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = sv2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")


# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 Random Forest:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

rf1 = RandomForestClassifier()

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
rf1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", rf1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = rf1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")




# THIS PART FOR CHECK RESIDENT 2 Random Forest:


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

rf2 = RandomForestClassifier()
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
rf2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", rf2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = rf2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")


# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 Logistic Regression:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data



print("\n.................................................................................\n")

lr1 = LogisticRegression()  

print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
lr1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", lr1.score(x1_test, y1_test))

print("\n.................................................................................\n")


y1_pred = lr1.predict(x1_test)

print(classification_report(y1_test, y1_pred))


precision, recall, fscore, support = score(y1_test, y1_pred)


print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")


# THIS PART FOR CHECK RESIDENT 2 Logistic Regression:



for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data



print("\n.................................................................................\n")

lr2 = LogisticRegression()  
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
lr2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", lr2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = lr2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")



# In[ ]:


# THIS PART FOR CHECK RESIDENT 1 GradientBoosting:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y1 = data.r1
data.drop(['r1'], axis=1 , inplace = True)
x1 = data


print("\n.................................................................................\n")

g1 = GradientBoostingClassifier()
print(x1)

print("\n.................................................................................\n")

x1_train, x1_test, y1_train, y1_test = train_test_split (x1,y1, test_size = 0.2, random_state = 42, stratify = y1)
g1.fit(x1_train,y1_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", g1.score(x1_test, y1_test))

print("\n.................................................................................\n")

y1_pred = g1.predict(x1_test)

print(classification_report(y1_test, y1_pred))
precision, recall, fscore, support = score(y1_test, y1_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")


# THIS PART FOR CHECK RESIDENT 2 GradientBoosting:


for i in range(10):
    globals()["url0" + str(i+1)] = 'https://raw.githubusercontent.com/seyedkazemi/aras/master/DAY_0'+str(i+1)+'.csv'
    globals()["df0" + str(i+1)] = pd.read_csv(globals()["url0" + str(i+1)])


w0 = df01.append(df02, ignore_index=True)
for j in range(7):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==6:
        data = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)


y2 = data.r2
data.drop(['r2'], axis=1 , inplace = True)
x2 = data


print("\n.................................................................................\n")

g2 = GradientBoostingClassifier()
print(x2)

print("\n.................................................................................\n")

x2_train, x2_test, y2_train, y2_test = train_test_split (x2,y2, test_size = 0.2, random_state = 42, stratify = y2)
g2.fit(x2_train,y2_train)
print("YOUR R2 MACHINE LEARNED WITH THIS ACCURACY : ", g2.score(x2_test, y2_test))

print("\n.........................................................................\n")


y2_pred = g2.predict(x2_test)

print(classification_report(y2_test, y2_pred))

precision, recall, fscore, support = score(y2_test, y2_pred)

print('precision: ',np.mean(precision))
print('recall: ',np.mean(recall))
print('fscore: ',np.mean(fscore))

print("---+++---+++---+++---+++---")

