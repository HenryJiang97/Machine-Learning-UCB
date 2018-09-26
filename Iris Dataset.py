
# coding: utf-8

# In[1]:


#Step1. Read the dataset

import pandas as pd
import numpy as np

iris = pd.read_csv('iris.csv')  # Read iris file
iris.iloc[[1,51,101]]           # Demonstrate the representative of three species


# In[2]:


#Step2. Split the dataset

from sklearn.cross_validation import train_test_split

#Split the dataset into two parts: Characteristic and Category (x = 'Characteristic', y = 'Categroy')
x, y = np.split(iris.values, (4,), axis=1)
y = y.reshape(len(iris),)  

# Split the dataset into train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=30)  


# In[3]:


#Step3. Classify using ID3 decision tree algorithm

from sklearn.tree import DecisionTreeClassifier

iris_tree = DecisionTreeClassifier(criterion = 'entropy')  # Classify using 'entropy' as metrics

iris_tree.fit(x_train, y_train)   # Train using training set

answer = iris_tree.predict(x)     # Predict all data

#Get the predicted results
answer_array = np.array([y, answer])
answer_mat = np.matrix(answer_array).T
result = pd.DataFrame(answer_mat)


# In[4]:


#Show results

result.columns = ['Real category', 'Predicted category']
result.iloc[[1,51,101]]


# In[5]:


#Step4. Visualize the decision tree

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.core.display import Image

# Set graphviz path
import os
os.environ["PATH"] += os.pathsep + 'B:/graphviz-2.38/release/bin/'

dot_data = StringIO()
export_graphviz(iris_tree, out_file=dot_data, filled=True,
               feature_names=iris.columns[:4],
               class_names=iris['Species'].unique(),
               rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[5]:


# Step5. Calculate precision, recall and f1-score  

from sklearn.metrics import classification_report

print("Validation of the result of training set data using decision tree：")
print(classification_report(y_test, iris_tree.predict(x_test)))
print(53*"-")

print("Validation of the result of all data using decision tree：")
print(classification_report(y, iris_tree.predict(x)))


# In[6]:


# Step6. 5-fold cross validation

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(iris_tree, x, y, cv=5)
cross = pd.DataFrame(scores)
cross.columns = ['Validation result']
cross.T

