
# House Price Prediction 

This is a Project to predict the house price using Machine Learning algorithms and exploratory data analysis.


## About The Dataset

Now to talk about I have taken the dataset from kaggle which is basically Boston house pricing and elaborating the same we have specific dependent variables with us and finally the price column which is the main aim to predict.
Some of the Variabes are:
   - Crime rate
   - TAX
   - Pollution
   - Age of Property etc...

So, these are the recorded variables in the dataset which are obsereved by the people who buy properties and therefore we will train our Model to predict the price based on all the variables that are present.


## Project details

The dataset comprises of the previous property buyers record in which the Price for buying a house is clearly given in final Price column.

Now we have splitted the dataset into training and test sets in which the accuracy score could be predicted so that we can see if our model is predicting right or wrong and what is the percentage of accuracy of our model.

So, we have applied the Linear Regression model in our dataset to predict the best possible Price according to the variables provided.

Finally, to test our model we have entered the input as self, which has predicted a perfect price for our house.
## Deployment

To see the results open the .ipynb file above.

```bash
  # House Price Prediction

## Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset

df = pd.read_csv('/content/BostonHousing.csv')

df.head(2)

df = df.rename(columns = {'medv': 'price'})

df.head(2)

## Exploratory Data Analysis

df.info()

df.describe()

### Identifing the correlation in the values

df.corr()

#### Scatter plot for some values just to check.

plt.scatter(df['crim'], df['price'])
plt.xlabel('crime rate')
plt.ylabel('price')

#### Regression line for the variables to check the positive correlation and negative correlation.

sns.regplot(x = "rm", y = "price", data=df)

sns.regplot(x='lstat', y='price', data=df)

## Preparation for training the dataset 

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

### Spliting into train and test data


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

#### Standardisation of dataset

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

## Model Training

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

print(regressor.coef_)

print(regressor.intercept_)

## Prediction of Data

reg_pred = regressor.predict(x_test)
reg_pred

### Scatter plot for DATA

plt.scatter(reg_pred, y_test)

### Finding the residuals

residual = reg_pred - y_test

#### Residual Plot

sns.displot(residual, kind='kde')

plt.scatter(residual, reg_pred)

####This is giving unifrom distribution

### Performance metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(reg_pred, y_test))
print(mean_squared_error(reg_pred, y_test))

### Predicting New data points

df.tail(2)

input_data = (0.10959, 0, 12.31, 0, 0.538, 6.58, 85.2, 3.0900, 1, 296, 20, 400, 4.98)
convert_data = np.asarray(input_data).reshape(1,-1)
print(regressor.predict(convert_data))

### Fantastic Prediction by our Model
```


## Conclusion

Now, eversince the model is trained perfectly therefore it can be a great helping hand for the people to predict if the property they are buying has same worth to what they are paying, and would surely help people to connect with the system and get their dream home at a desired price.
# Hi, I'm Avichal Srivastava ! 👋

You can reach out to me at: srivastavaavichal007@gmail.com

LinkedIn: www.linkedin.com/in/avichal-srivastava-186865187


## 🚀 About Me

I'm a Mechanical Engineer by education, and love to work with data, I eventually started my coding journey for one of my Drone project, wherein I realized that it is something which makes me feel happy in doing, then I planed ahead to move in the Buiness Analyst or Data Analyst domain. The reason for choosing these domains is because I love maths a lot and all the Machine Learning algorithms are completely based on mathematical intution, So this was about me.

Hope! You liked it, and its just a beginning, many more to come, till then Happy Analysing!!

