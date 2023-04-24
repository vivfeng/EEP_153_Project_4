#!/usr/bin/env python
# coding: utf-8

# # Configuration Code

# In[1]:


get_ipython().run_line_magic('pip', 'install gspread_pandas')
get_ipython().run_line_magic('pip', 'install fooddatacentral')
get_ipython().run_line_magic('pip', 'install pint')
get_ipython().run_line_magic('pip', 'install cufflinks')
get_ipython().run_line_magic('pip', 'install CFEDemands')
get_ipython().run_line_magic('pip', 'install eep153_tools')
get_ipython().run_line_magic('pip', 'install gnupg')


from  scipy.optimize import linprog as lp
import numpy as np
import warnings
import pandas as pd
import eep153_tools
from eep153_tools.sheets import read_sheets
import fooddatacentral as fdc
import cufflinks as cf


from cfe import Regression

cf.go_offline()


# # (A) Choice of Population, with supporting expenditure
# 

# We chose to analyze the Ugandan popultion of males and females 19-30 
# 

# Ugandan Expenditures of 2019-20 

# In[2]:


Uganda_Data = '1yVLriVpo7KGUXvR3hq_n53XpXlD5NmLaH1oOMZyV0gQ'

x = read_sheets(Uganda_Data,sheet='Expenditures (2019-20)') 
x.columns.name = 'j'


# Ugandan Household characteristics 

# In[3]:


d = read_sheets(Uganda_Data,sheet="HH Characteristics") 
d.columns.name = 'k'


# In[4]:


x = x.groupby('j',axis=1).sum() #reducing duplicate columns
x = x.replace(0,np.nan) #reducing nulls
y = np.log(x.set_index(['i','t','m'])) #log of expenditure 
d.set_index(['i','t','m'],inplace=True) #specific labels for the axis


use = y.index.intersection(d.index)
y = y.loc[use,:]
d = d.loc[use,:]


#Filtering it down to our population of interest (M,F 19-30) 
b = read_sheets(Uganda_Data,sheet='RDI')
b = b.set_index('n')



# # (A) Estimate Demand System
# 

# In[5]:


from cfe.estimation import drop_columns_wo_covariance
y = drop_columns_wo_covariance(y,min_obs=30)
use = y.index.intersection(d.index)
y = y.loc[use,:]
d = d.loc[use,:]

#y is log expednitures on food j by household i at a particular time
y = y.stack()
d = d.stack()
assert y.index.names == ['i','t','m','j']
assert d.index.names == ['i','t','m','k']

#setting up the regression
result = Regression(y=y,d=d)
#predicting expenditures
result.predicted_expenditures()


# Comparing Predicted Log Expenditures with Actual Expenditures

# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
df = pd.DataFrame({'y':y,'yhat':result.get_predicted_log_expenditures()})
df.plot.scatter(x='yhat',y='y', title = 'Log Expenditures vs. Actual Expenditures')


# Demand and Household Composition
# Relative to the average consumption, the characteristics of age and sex affect the demand of the household in this factor. 

# In[7]:


result.gamma


# # (B) Nutritional Content of Different Foods
# 

# In[9]:


food_nutrient = pd.read_excel("Uganda.xlsx", sheet_name = "FCT")
food_nutrient


# # (B) Nutritional Adequacy of Diet
# 

# In[10]:


expenditure = pd.read_excel("Uganda.xlsx", sheet_name = "Expenditures (2019-20)")
expenditure


# In[11]:


price = pd.read_excel("Uganda.xlsx", sheet_name = "Prices")
price = price[price["t"] == "2019-20"]
price


# In[12]:


foods = ['Beans', 'Beef', 'Beer', 'Biscuits', 'Bongo', 'Bread',
       'Butter, etc.', 'Cabbages', 'Cake', 'Cassava', 'Cassava (flour)',
       'Chapati', 'Cheese', 'Chicken', 'Cigarettes', 'Coffee', 'Cooking Oil',
       'Cornflakes', 'Dodo', 'Donut', 'Eggs', 'Fish (dried)', 'Fish (fresh)',
       'Garlic', 'Ghee', 'Ginger', 'Goat', 'Ground Nuts', 'Honey', 'Ice Cream',
       'Infant Formula', 'Irish Potatoes', 'Jackfruit', 'Jam/Marmalade',
       'Kabalagala', 'Macaroni/Spaghetti', 'Maize', 'Mangos', 'Matoke',
       'Milk (fresh)', 'Milk (powdered)', 'Millet', 'Onions', 'Oranges',
       'Other Alcohol', 'Other Drinks', 'Other Fruits', 'Other Juice',
       'Other Meat', 'Other Spices', 'Other Tobacco', 'Other Veg.',
       'Passion Fruits', 'Peas', 'Plantains', 'Pork', 'Rice', 'Salt', 'Samosa',
       'Sim Sim', 'Soda', 'Sorghum', 'Soybean', 'Sugar', 'Sugarcane',
       'Sweet Bananas', 'Sweet Potatoes', 'Tea', 'Tomatoes', 'Waragi', 'Water',
       'Wheat (flour)', 'Yam', 'Yogurt']
expenditure_and_price = expenditure.merge(price, how = "left", on = "m")
expenditure_and_price


# In[13]:


for food in foods:
    exp = str(food) + "_x"
    price = str(food) + "_y"
    expenditure_and_price[food] = expenditure_and_price[exp]/expenditure_and_price[price]
household_consumption = expenditure_and_price[foods].fillna(0)
household_consumption


# In[14]:


#household dempgrahics 
household = pd.read_excel("Uganda.xlsx", sheet_name = "HH Characteristics")
household_19_20 = household[household["t"] == "2019-20"]
household_19_20 


# In[15]:


household_consumption['i'] = expenditure["i"]
household_consumption = household_consumption.merge(household_19_20, how = "left", on = "i")
household_consumption


# In[16]:


RDI = pd.read_excel("Uganda.xlsx", sheet_name = "RDI")
RDI


# In[17]:


x = household_consumption[['F 00-03', 'F 04-08', 'F 09-13', 'F 14-18', 'F 19-30',
       'F 31-50', 'F 51+', 'M 00-03', 'M 04-08', 'M 09-13', 'M 14-18',
       'M 19-30', 'M 31-50', 'M 51+']]
x


# In[18]:


y = RDI[['F 00-03', 'F 04-08', 'F 09-13', 'F 14-18', 'F 19-30',
       'F 31-50', 'F 51+', 'M 00-03', 'M 04-08', 'M 09-13', 'M 14-18',
       'M 19-30', 'M 31-50', 'M 51+']].transpose()
y


# In[19]:


required_nutrients_household = x@y
required_nutrients_household.columns = RDI["n"]
required_nutrients_household


# In[20]:


food_consumed = ['Beans', 'Beef', 'Beer', 'Biscuits', 'Bongo',
       'Bread', 'Butter, etc.', 'Cabbages', 'Cake', 'Cassava',
       'Cassava (flour)', 'Chapati', 'Cheese', 'Chicken', 'Cigarettes',
       'Coffee', 'Cooking Oil', 'Cornflakes', 'Dodo', 'Donut', 'Eggs',
       'Fish (dried)', 'Fish (fresh)', 'Garlic', 'Ghee', 'Ginger', 'Goat',
       'Ground Nuts', 'Honey', 'Ice Cream', 'Infant Formula', 'Irish Potatoes',
       'Jackfruit', 'Jam/Marmalade', 'Kabalagala', 'Macaroni/Spaghetti',
       'Maize', 'Mangos', 'Matoke', 'Milk (fresh)', 'Milk (powdered)',
       'Millet', 'Onions', 'Oranges', 'Other Alcohol', 'Other Drinks',
       'Other Fruits', 'Other Juice', 'Other Meat', 'Other Spices',
       'Other Tobacco', 'Other Veg.', 'Passion Fruits', 'Peas', 'Plantains',
       'Pork', 'Rice', 'Salt', 'Samosa', 'Sim Sim', 'Soda', 'Sorghum',
       'Soybean', 'Sugar', 'Sugarcane', 'Sweet Bananas', 'Sweet Potatoes',
       'Tea', 'Tomatoes', 'Waragi', 'Water', 'Wheat (flour)', 'Yam', 'Yogurt']


# In[21]:


food_nutrient = food_nutrient[food_nutrient['j'].isin(food_consumed)]
x_2 = household_consumption[food_nutrient['j']]
x_2 = x_2.fillna(0)
x_2


# In[22]:


y_2 = food_nutrient.iloc[:,1:].set_index(food_nutrient['j'])
y_2


# In[23]:


consumed_nutrients = x_2@y_2
consumed_nutrients


# In[24]:


required_nutrients_household = required_nutrients_household[consumed_nutrients.columns].fillna(0)
required_nutrients_household*7


# In[25]:


proportions = []
for nutrient in consumed_nutrients.columns:
    proportion = required_nutrients_household[nutrient]/consumed_nutrients[nutrient]
    proportions.append(proportion)


# In[26]:


nutritional_adequancy = pd.DataFrame(proportions).transpose()
nutritional_adequancy.replace(np.inf, 0,inplace=True)
nutritional_adequancy.fillna(0)
nutritional_adequancy = nutritional_adequancy.drop(["Vitamin B-12"], axis = 1)
nutritional_adequancy


# In[27]:


import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.bar(nutritional_adequancy.mean().index,nutritional_adequancy.mean() )
plt.axhline(y = 1, color = 'r', linestyle = '-')
plt.title("Nutritional adequacy of hosuehold diet (Uganda 2019 - 2020)")
plt.xticks(fontsize= 9)
plt.ylabel("Ratio of recomended nutrients over actual consumption")


# In[ ]:





# In[ ]:




