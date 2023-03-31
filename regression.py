# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# %%
###! pip install scikit-learn

# %%
###! pip install klib

# %%
import klib

# %%
import pandas as pd

# %%
dataframe = pd.read_csv("listings (1).csv", encoding='utf-8')

# %%
dataframe.head(2)

# %%
dataframe.info()

# %%
dataframe.isnull().sum()

# %%
dataframe.drop(columns = {"neighbourhood_group"}, inplace=True)
dataframe.drop(columns = {"last_review"}, inplace=True)

# %%
dataframe.drop(columns = {"reviews_per_month"}, inplace=True)
dataframe.drop(columns = {"license"}, inplace=True)

# %%
print(dataframe.info())

# %%
dataframe = dataframe.fillna(method ='pad')

# %%
dataframe.isnull().sum()

# %%
dataframe = klib.convert_datatypes(dataframe) 

# %%
print(dataframe.info())

# %%
dataframe['room_type'] = dataframe['room_type'].astype('category').cat.codes

# %%
dataframe['host_name'] = dataframe['host_name'].astype('category').cat.codes
dataframe['name'] = dataframe['name'].astype('category').cat.codes

# %%
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

# %%
from sklearn.model_selection import train_test_split

training_data, testing_data = train_test_split(dataframe, test_size=0.2, random_state=25)

# %%
print(training_data.shape, testing_data.shape)

# %%
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

# %%
model = SelectFromModel(Lasso(alpha=0.005,random_state=0))

# %%
X_train = training_data.drop(columns = "price")
X_test = testing_data.drop(columns = "price")

# %%
y_train = training_data["price"]
y_test = testing_data["price"]

# %%
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
model.fit(X_train, y_train)

# %%
model.get_support()

# %%
selected_features = X_train.columns[(model.get_support())]

# %%
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# %%
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

# %%
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# %%
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# %%
print(X_train.columns,y_train.columns)
print(X_test.columns,y_test.columns)

# %%
model_rf = RandomForestRegressor()

# %%
model_rf.fit(X_train, y_train)

# %%
y_pred_rf = model_rf.predict(X_test)

# %%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# %%
print("Mean Absolute Error",mean_absolute_error(y_test,y_pred_rf))
print("Mean Squared Error",mean_squared_error(y_test,y_pred_rf))
print("Mean Absolute Percentage Error",mean_absolute_percentage_error(y_test,y_pred_rf))

# %%
from sklearn.ensemble import GradientBoostingRegressor

# %%
model_gb = GradientBoostingRegressor()

# %%
model_gb.fit(X_train, y_train)

# %%
y_pred_gb = model_gb.predict(X_test)

# %%
print("Mean Absolute Error",mean_absolute_error(y_test,y_pred_gb))
print("Mean Squared Error",mean_squared_error(y_test,y_pred_gb))
print("Mean Absolute Percentage Error",mean_absolute_percentage_error(y_test,y_pred_gb))

# %%



