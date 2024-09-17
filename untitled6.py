

Original file is located at
    https://colab.research.google.com/drive/1AN748JZM8oxAitfi-hhKdiP7MTW7reKV

**Online Food Order Prediction using Python**

Now let’s start with the task of online food order prediction with machine learning. You can download the dataset I am using for this task from here. Let’s start with importing the necessary Python libraries and the dataset:
"""

import numpy as np
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

data = pd.read_csv("onlinefoods.csv")
print(data.head())
print(data.info())

"""So the dataset contains information like:

the age of the customer
marital status of the customer
occupation of the customer
monthly income of the customer
educational qualification of the customer
family size of the customer
latitude and longitude of the location of the customer
pin code of the residence of the customer
did the customer order again (Output)
Feedback of the last order (Positive or Negative)

**Let’s have a look at the information about all the columns in the dataset:**
"""

print(data.info())

plt.figure(figsize=(10,5))
plt.title("Online food Order Decisions Based on the Age of the Customer")
sns.histplot(x="Age",hue="Output",data=data)
plt.show()

"""We can see that the age group of 22-25 ordered the food often again.It also means this age group is the target of online food delivary companies.

**Now lets have a look at the online food order decisions based on the size of the family of the cutomer:**
"""

plt.figure(figsize=(10,5))
plt.title("Online food Order Decisions Based on the Size of the Family")
sns.histplot(x="Family size",hue="Output",data=data)
plt.show()

"""Families with 2 and 3 members are ordering food often.These can be roommates,couples,or a family of three

**Lets create a dataset of all the customers who ordered the food again:**
"""

buying_again_order=data.query("Output=='Yes'")
print(buying_again_order.head())

"""Now lets have a look at the gender column.
Lets find who orders food more online:
"""

gender=buying_again_order["Gender"].value_counts()
label=gender.index
counts=gender.values
colors=['blue','green']
fig =go.Figure(data=[go.Pie(labels=label,values=counts)])
fig.update_layout(title_text="Who Orders Food More:Male VS Female")
fig.update_traces(hoverinfo='label+percent',textinfo='value',textfont_size=30,marker=dict(colors=colors,line=dict(color='black',width=3)))
fig.show()

"""According to the dataset,male customers are ordering more compared the females.

**lets have a look at the marital status of the customers who ordered again:**
"""

marital=buying_again_order["Marital Status"].value_counts()
label=marital.index
counts=marital.values
colors=['blue','green']
fig =go.Figure(data=[go.Pie(labels=label,values=counts)])
fig.update_layout(title_text="Who Orders Food More:Married VS Single")
fig.update_traces(hoverinfo='label+percent',textinfo='value',textfont_size=30,marker=dict(colors=colors,line=dict(color='black',width=3)))
fig.show()

"""According to the above figure, 76.1% of the frequent customers are singles

**Now let’s have a look at what’s the income group of the customers who ordered the food again:**
"""

income = buying_again_order["Monthly Income"].value_counts()
label = income.index
counts = income.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Which Income Group Orders Food Online More')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

"""According to the above figure, 54% of the customers don’t fall under any income group. They can be housewives or students

**Now let’s prepare the data for the task of training a machine learning model. Here I will convert all the categorical features into numerical values:**
"""

data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Marital Status"] = data["Marital Status"].map({"Married": 2,
                                                     "Single": 1,
                                                     "Prefer not to say": 0})
data["Occupation"] = data["Occupation"].map({"Student": 1,
                                             "Employee": 2,
                                             "Self Employeed": 3,
                                             "House wife": 4})
data["Educational Qualifications"] = data["Educational Qualifications"].map({"Graduate": 1,
                                                                             "Post Graduate": 2,
                                                                             "Ph.D": 3, "School": 4,
                                                                             "Uneducated": 5})
data["Monthly Income"] = data["Monthly Income"].map({"No Income": 0,
                                                     "25001 to 50000": 5000,
                                                     "More than 50000": 7000,
                                                     "10001 to 25000": 25000,
                                                     "Below Rs.10000": 10000})
data["Feedback"] = data["Feedback"].map({"Positive": 1, "Negative ": 0})
print(data.head())

"""# **Online Food Order Prediction Model**

**Now let’s train a machine learning model to predict whether a customer will order again or not. I will start by splitting the data into training and test sets:**
"""

#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["Age", "Gender", "Marital Status", "Occupation",
                   "Monthly Income", "Educational Qualifications",
                   "Family size", "Pin code", "Feedback"]])
y = np.array(data[["Output"]])

"""**Now let’s train the machine learning model:**"""

from sklearn.ensemble import RandomForestClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.10,
                                                random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
