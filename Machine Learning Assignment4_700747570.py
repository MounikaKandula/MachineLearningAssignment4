#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries to work with Tabular data and also to implement algorithms

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")


# In[2]:


#1. Read the provided CSV file ‘data.csv’. https://drive.google.com/drive/folders/1h8C3mLsso-R-sIOLsvoYwPLzy2fJ4IOF?usp
df = pd.read_csv("/Users/mounikakandula/Downloads/ML Assignment/data.csv")
print(df.head())


# In[3]:


#2. Show the basic statistical description about the data.
print(df.describe())


# In[4]:


#3. Check if the data has null values.
df.isnull().any()


# In[5]:


#Replace the null values with the mean
df.fillna(df.mean(), inplace=True)
df.isnull().any()


# In[6]:


#4. Select at least two columns and aggregate the data using: min, max, count, mean.
df.agg({'Duration':['min','max','count','mean'],'Pulse':['min','max','count','mean']})


# In[7]:


#5. Filter the dataframe to select the rows with calories values between 500 and 1000.
df.loc[(df['Calories']>500)&(df['Calories']<1000)]


# In[8]:


#6. Filter the dataframe to select the rows with calories values > 500 and pulse < 100.
df.loc[(df['Calories']>500)&(df['Pulse']<100)]


# In[9]:


#7. Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”.
df_modified = df[['Duration','Pulse','Calories']]
df_modified.head()


# In[10]:


#8. Delete the “Maxpulse” column from the main df dataframe
del df['Maxpulse']


# In[11]:


df.head()


# In[12]:


df.dtypes


# In[13]:


#9. Convert the datatype of Calories column to int datatype.

df['Calories'] = df['Calories'].astype(np.int64)
df.dtypes


# In[14]:


#10. Using pandas create a scatter plot for the two columns (Duration and Calories).

df.plot.scatter(x='Duration',y='Calories',c='blue')


# In[15]:


import warnings
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
# Supresswarnings
warnings.filterwarnings("ignore")


# In[16]:


df=pd.read_csv("/Users/mounikakandula/Downloads/ML Assignment/Dataset/train.csv")


# In[17]:


df.head()


# In[18]:


le = preprocessing.LabelEncoder()
df['Sex'] = le.fit_transform(df.Sex.values)
df['Survived'].corr(df['Sex'])


# In[19]:


## a. Do you think we should keep this feature?

## A negative (inverse) correlation occurs when the correlation coefficient is less than 0. This is an indication that both variables move in the opposite direction. In short, any reading between 0 and -1 means that the two securities move in opposite directions. If one variable increases, the other variable decreases with the same magnitude (and vice versa). However, the degree to which two securities are negatively correlated might vary over time (and they are almost never exactly correlated all the time). Removing a correlated feature does not make any difference in the outcome of the model. It is always better to remove the highly correlated features first and then least correlated ones.


# In[20]:


des=df.corr()
df.corr().style.background_gradient(cmap="pink")


# In[21]:


sns.barplot(data=des) #BarPlot Visualization for above dataset


# In[22]:


sns.histplot(data=des) #Histogram Visualization for above dataset


# In[23]:


train_raw = pd.read_csv("/Users/mounikakandula/Downloads/ML Assignment/Dataset/train.csv")
test_raw = pd.read_csv("/Users/mounikakandula/Downloads/ML Assignment/Dataset/test.csv")

# Join data to analyse and process the set as one.
train_raw['train'] = 1
test_raw['train'] = 0
df = train_raw.append(test_raw, sort=False)



features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

df = df[features + [target] + ['train']]
# Categorical values need to be transformed into numeric.
df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df.query('train == 1')
test = df.query('train == 0')
      


# In[24]:


# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values

train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)


# In[25]:


from sklearn.model_selection import train_test_split, cross_validate

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)


# In[26]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
# Suppress warnings
warnings.filterwarnings("ignore")


# In[27]:


classifier = GaussianNB()

classifier.fit(X_train, Y_train)


# In[28]:


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[29]:


glass=pd.read_csv("/Users/mounikakandula/Downloads/ML Assignment/Dataset/glass.csv") 


# In[30]:


glass.head()


# In[31]:


des=glass.corr()
glass.corr().style.background_gradient(cmap="pink")


# In[32]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass[target],test_size=0.2, random_state=1)


# In[33]:


classifier = GaussianNB()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('\naccuracy is',accuracy_score(Y_val, y_pred))


# In[34]:


from sklearn.svm import SVC, LinearSVC
classifier = LinearSVC()

classifier.fit(X_train, Y_train)


# In[35]:


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('\naccuracy is',accuracy_score(Y_val, y_pred))


# In[36]:


sns.heatmap(data=glass) #HeatMap Visualization for above dataset


# In[37]:


sns.scatterplot(data=glass)  #ScatterPlot Visualization for above dataset


# In[ ]:




