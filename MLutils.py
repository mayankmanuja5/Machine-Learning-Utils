
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import xgboost as xgb
import MLutils as myutils


# In[2]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[3]:


def standardscale(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()                   #Instantiate the scaler
    scaled_X_train = scaler.fit_transform(X)    #Fit and transform the data
    X=scaled_X_train
    return X


# In[4]:


from sklearn.preprocessing import LabelEncoder
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
                df[feature] = le.fit_transform(df[feature])
        return df

def split(X,y,splitratio):
    trainsize=int(X.shape[0]*splitratio)
    xtrain=X[:trainsize]
    ytrain=y[:trainsize]
    xtest=X[trainsize:]
    ytest=y[trainsize:]
    return xtrain,ytrain,xtest,ytest

def get_accuracy(y_pred,yp2):
    right=0
    for i in range(len(y_pred)):
        if(y_pred[i]==yp2[i]):
            right+=1
    return (100*right/float(len(y_pred)))


# In[5]:


def plot_corr(ds):
    corr = ds.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[6]:


# Importing PCA = principal component analysis
def pca(X,comp):
    X=standardscale(X)
    from sklearn.decomposition import PCA 
    pca = PCA(n_components = comp) 
    pca.fit(X) 
    X = pca.transform(X) 
    return X


# In[7]:



#convert to list
def converttolist(ytest):
    yp2=[]
    for i in ytest:
        yp2.append(i)
    return yp2


# In[8]:


#naive bayes
def naiveBayes(X,y,xtest):
    print("naive bayes")
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(X, y).predict(xtest)
    return y_pred


# In[9]:


#Decision Tree Classifier , run multiple times to get diff accuracy 
def decisionTree(X,y,xtest):
    from sklearn import tree
    print("dec-tree")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    y_pred=clf.predict(xtest)
    return y_pred


# In[10]:


#adabooster , n estimators
def adaBooster(X,y,xtest):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    print("adabooster")
    bdt = AdaBoostClassifier(LogisticRegression(),
                             algorithm="SAMME",
                             n_estimators=100)
    bdt.fit(X,y)
    y_pred=bdt.predict(xtest)
    return y_pred



# In[11]:


#Random forest classifier
def randomForest(X,y,xtest):
    print("randome forest")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(X, y)
    y_pred=clf.predict(xtest)
    return y_pred


# In[12]:


#Gradient Boosting Classifier
def gradientBoost(X,y,xtest):
    from sklearn.ensemble import GradientBoostingClassifier
    print("Gradient boost started")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=1,random_state=0)
    clf.fit(X, y)
    y_pred=clf.predict(xtest)
    return y_pred


# In[13]:


def logisticRegression(X,y,xtest):
    from sklearn.linear_model import LogisticRegression
    print("Logistic Regression")
    clf = LogisticRegression().fit(X, y)
    y_pred=clf.predict(xtest)
    return y_pred


# In[14]:


def logisticRegressionCV(X,y,xtest):
    from sklearn.linear_model import LogisticRegressionCV
    print("Logistic Regression CV")
    clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
    y_pred=clf.predict(xtest)
    return y_pred


# In[15]:


def svm(X,y,xtest):
    print("Support-Vector-Machine")
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', gamma=0.7,max_iter=10000)
    clf.fit(X, y) 
    y_pred=clf.predict(xtest)
    return y_pred


# In[16]:


def linearsvm(X,y,xtest):
    print("Linear Support-Vector-Machine")
    from sklearn.svm import LinearSVC
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X, y)
    y_pred=clf.predict(xtest)
    return y_pred


# In[17]:


def Nusvc(X,y,xtest):
    print("Nu Support-Vector-Machine")
    from sklearn.svm import NuSVC
    clf = NuSVC(random_state=0,gamma='auto')
    clf.fit(X, y) 
    y_pred=clf.predict(xtest)
    return y_pred


# In[18]:


def knn(X,y,xtest,k=None):
    if not k:
        k=.02*len(xtest)
    print("knn")
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y) 
    y_pred=neigh.predict(xtest)
    return y_pred


# In[19]:


def mlp(X,y,xtest):
    print("mlp")
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=100, verbose=10, random_state=7)
    mlp.fit(X, y) 
    y_pred=mlp.predict(xtest)
    return y_pred


# In[20]:


#returns all subsets of s
def subsets(s):
    sets = []
    for i in range(1 << len(s)):
        subset = [s[bit] for bit in range(len(s)) if is_bit_set(i, bit)]
        sets.append(subset)
    return sets

def is_bit_set(num, bit):
    return num & (1 << bit) > 0


# In[21]:


def callall(X,y,xtest,ytest):
    flist=["naiveBayes","decisionTree","adaBooster","randomForest","gradientBoost","logisticRegression","logisticRegressionCV","svm","linearsvm","Nusvc","mlp"]
    print("calling")
    print(flist)
    for f in flist:
        method_to_call = getattr(myutils,f)
        ypred=method_to_call(X,y,xtest)
        print(get_accuracy(ypred,ytest))
        

