#!/usr/bin/env python
# coding: utf-8

# # Graduate Rotational Internship Program
# # Task 3
# # Exploratory Data Analysis -Retail
# **Mansi Mayuri**

# ### Importing Esential Librery for Vsualization and Data analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading .csv Dataset and printing top 5 column

# In[2]:


df =pd.read_csv('samplesuperstore.csv')


# In[3]:


df.head()


# ### Check for any null value in Dataset

# In[4]:


df.isnull().sum()


# ### Check for Duplicate value

# In[5]:


df.duplicated().sum()


# ### Drop all possible Duplicate value

# In[6]:


df.drop_duplicates(keep='first',inplace=True)
df


# ### Printing Datatypes of all columns and name of columns 

# In[7]:


df.dtypes


# In[8]:


df.columns


# In[9]:


np.shape(df)


# #### By finding uniqueness of  Country we got  only United State is present in dataset so we can simpaly drop this column. Because this is common for all the rows.

# In[10]:


df['Country'].unique()


# In[11]:


df.drop("Country", axis = 1, inplace=True)


# In[12]:


df.columns
np.shape(df)


# In[13]:


df.head()


# #### By describe() method we got the basic statatics of dataset like min, standerd deviation,count etc

# In[14]:


df.describe()


# ### By corr() method we find that how much each label is  related to each other 
# ***Its value varies from -1 to 1***
# **+1 means both the label has strong correlation and it decrease from +1 to -1 and simultaneously correlation in specific label also decreases**

# In[15]:


cor =df.corr()
cor


# #### With help of heatmap we can plot the correlation of variables which is easy to analyze
# * In following map we can analyze that different colur represent different valu of correlation. It is one for same variable and represented by cream color and least correlated is sales v/s discount and this is represented by nevy blue color. How the color varies with different value can be analyzed by given colorbar.

# In[16]:


sns.heatmap(cor,annot =True)


# #### In this bar plot we are comparing different categories and their respective Sales count . By analyzing this plot we can see that maximum sales occur in Technology category after that Furniture and least in Office supplies.

# In[17]:


sns.barplot(df['Category'],df['Sales'])


# In[18]:


sns.jointplot(x='Sales',y= 'Quantity', data=df)


# ### Here we are finding all type of discount the shop is providing and than we are finding maximum and minimum value of the discount.
# 

# In[19]:


Dis =df['Discount'].unique()
Dis
mx =max(df['Discount'])
mn =min(df['Discount'])
print('max value of discount is mx',mx)
print('min value of Discount is ',mn)


# In[20]:


max_dis=df[df['Discount'] ==mx]
min_dis =df[df['Discount']==mn]


# ### Here we are ploting regplot to find relation between sales and profit at  particular discount .By analyzing the plot at min and max value of discount we found that at max value of discount (0.8) we are getting linear relationship but with negative slope that shows that profit will decrease with increase in sales at this discount value . So This is not reliable.

# In[21]:


sns.regplot(x ='Sales',y ='Profit',data =max_dis)


# ### Here we are ploting regplot to find relation between sales and profit at  particular discount .By analyzing the plot at min and max value of discount we found that at min value of discount (0.0) we are getting linear relationship and with positive slope that shows that profit will increase with increase in sales at this discount value . So This is reliable condition.

# In[22]:


sns.regplot(x ='Sales',y='Profit',data =min_dis)


# #### Here we are finding the stuff count of different catrgory and sub-category  which have been sold out.

# In[23]:


sub_cat =df['Sub-Category'].value_counts()
cat =df['Category'].value_counts()
print('sub catogery are',sub_cat)
print('catogary are',cat)


# ### these two following count plot is showing count of category and sub category its for compare purpose

# In[24]:


sns.countplot(x ='Category',data =df)


# In[25]:


plt.figure(figsize=(15,8))
sns.countplot(x ='Sub-Category',data =df)


# ### This bar plot shows the relation between sales of different sub category .By analyzing this plot we can conclude that copiers have maximum saling ratio and this is least for Fasteners than art type stuff.

# In[26]:


plt.figure(figsize=(18,8))
sns.barplot(df['Sub-Category'],df['Sales'],data =df)


# 

# #### This bar plot is giving relation between category and sub-category of seles. So we can conclude that most of sub-category related with Office supplies and for sub-category involved with Technology. By this and analyzing profit vs sub-category graph we can conclude is this benificial or not

# In[27]:


plt.figure(figsize=(18,8))
plt.bar('Sub-Category','Category',data =df)


# #### This bar plot is among 3 variables sub-category,sales and profit means by analyzing we can find the profit and sales of each sub-category .In graph we can see that for phones profit is not good according to its sales but in case of copiers we can see that less no of sales is giving more  profit.
#                                          So we can increase  its sales in order to get more profit

# In[28]:



df.groupby(['Sub-Category'])['Sales','Profit'].agg(['sum']).plot.bar()
plt.show()


# #### This bar plot is among 3 variables category,sales and profit means by analyzing we can find the profit and sales of each category .In graph we can see that for Technology category  profit is good as its sales is also maximm .

# In[29]:


df.groupby(['Category'])['Sales','Profit'].agg(['sum']).plot.bar()
plt.show()


# In[30]:


df['Profit'].sort_values()


# In[31]:


state_profit=df.groupby(['State']).Profit.mean().reset_index()
state_profit


# In[ ]:





# In[32]:


state_sale =df.groupby(['State'])['Sales'].mean().reset_index()
state_sale


# ### Here we are finding the dataframe which consist of State with their sales number and profit 

# In[33]:



result =pd.merge(state_profit,state_sale)
result


# ### This groupby plot is among state and their respective total sales vs total profit .So by analyzing this graph we conclude that  Wyoming consist maximum no of sales while max profit happens in vermont

# In[34]:



result.groupby(['State'])['Sales','Profit'].agg(['sum']).plot.bar()
plt.show()


# In[35]:


df['Discount']=df['Discount']*100


# #### This groupby bar plot is giving sales and profit with respect of discount in each region .So we can see that in west region discount is not more but sales and profit is better respect to others region .In central region discount is maximum but sales and profit both are less

# In[36]:



df.groupby(['Region'])['Sales','Profit','Discount'].agg(['sum']).plot.bar()
plt.show()


# ### This groupby bar plot is showing sales ,profit vs respective discount in each shipping mode .Like we can see that in standard class discount is maximum and sales and profit also more .we can also cnclude that in standerd class profit and sales are totally depens upon discount.
# 

# In[37]:


df.groupby(['Ship Mode'])['Sales','Profit','Discount'].agg(['sum']).plot.bar()
plt.show()


# #### Sorting state name accordiing to increasing profit and area plot of profit vs sales

# In[38]:


results =result.sort_values(by ='Profit',ascending =False)
results.head(15)


# In[39]:


results.plot.area(x ='Profit',y ='Sales')
plt.show()


# In[40]:


import folium
world_map =folium.Map(zoom_start =4)


# In[ ]:


world_geo =r'usa_state.json'


# In[ ]:


world_map.choropleth(
    geo_data =world_geo,
    data =df,
    columns =['State','Sales'],
    key='feature.properties.states',
    fill_color='YlGnBu',
    fill_opacity=0.7, line_opacity=0.2,
    legend_name='hist_indicator')

world_map


# In[ ]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# ## Building a ML model to predict the values of preferable Discount to make more profit.

# In[ ]:


import numpy as np
import pandas as pd
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics


# In[ ]:


df =pd.read_csv('SampleSuperstore.csv')


# In[ ]:


df.drop(['Country'],axis =1,inplace =True)
df['Discount']=df['Discount']*100


# In[ ]:


feature_cols = ['Sales', 'Profit','Quantity']
X = df[feature_cols]

y =df.Discount


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[ ]:


clf = DecisionTreeClassifier()
history= clf.fit(X_train,y_train)


# In[ ]:


y_pred = clf.predict(X_test)
np.mean(y_pred)


# In[ ]:


np.mean(y_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# #  **Thank you**
