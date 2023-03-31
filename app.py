
# Core Pkgs

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import klib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error



st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Aiplane Ticket Regression Web App")
st.subheader("Creator : Debmalya Ray")
st.sidebar.title("Air Ticket Price Regression")


def main():
	"""Semi Automated ML App with Streamlit """

	activities = ["EDA","Plots","Machine_Learning"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data, encoding='utf-8')
			df = klib.convert_datatypes(df)
			st.dataframe(df.head())
			df = df.fillna(method ='pad')
	
			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Correlation Plot(Matplotlib)"):
				plt.matshow(df.corr())
				st.pyplot()

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()



	elif choice == 'Plots':
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data, encoding='utf-8')
			st.write("The Dataset is :")
			st.dataframe(df.head(10))


			if st.checkbox("Show descriptive data"):
				st.write(df.head(10))
				st.pyplot()
		
			# Customizable Plot
 
			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()



	elif choice == 'Machine_Learning':

		st.subheader("Machine Learning (ML)")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data, encoding='utf-8')
			df = klib.data_cleaning(df)
			df = klib.convert_datatypes(df)
			st.write("The Columns Of The DataFrame :")
			st.dataframe(df.columns)
			#st.dataframe(df.head(2))
			df['room_type'] = df['room_type'].astype('category').cat.codes
			df['host_name'] = df['host_name'].astype('category').cat.codes
			df['name'] = df['name'].astype('category').cat.codes
			df['last_review'] = df['last_review'].astype('category').cat.codes
			df = df.fillna(method ='pad')
			X = df.drop(columns = "price")
			Y = df["price"]

			X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

			st.sidebar.subheader("Choose Regressor")
			regressor = st.sidebar.selectbox("Regressor", ("Random Forest", "Gradient Boosting", "Ada Boosting"))
		#	if st.sidebar.button("Classify", key='classify'):

			if regressor == 'Random Forest':
				st.sidebar.subheader("Model Hyper-parameters")
				max_depth = st.sidebar.number_input("The maximum depth of the tree", 5, 20, step=1, key='max_depth')
				bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

				if st.sidebar.button("Regressor", key='regressor') :
					st.subheader("Random Forest Results")
					model = RandomForestRegressor(max_depth=max_depth, bootstrap=bootstrap)
					model.fit(X_train,Y_train)
					#accuracy = model.score(X_test, Y_test)
					y_pred = model.predict(X_test)
					mae2 = mean_absolute_error(Y_test, y_pred)
					mse2 = mean_squared_error(Y_test, y_pred)
					y_result = pd.DataFrame(y_pred)
					y_result.rename(columns={0:"Predict"}, inplace=True)
					#d =  {0: '0 - Unassigned', 1: '1 - Low', 2: '2 - Medium', 3: '3 - High'}
					#y_result = y_result["Predict"].map(d)
					#accuracy = model.score(X_test, Y_test)

					plt.figure(figsize=(12,6), dpi=80, facecolor='green')
					#bw = y_result.value_counts()

					st.write("Mean Absolute Error : ", mae2)
					st.write("Mean Squared Error : ", mse2)
					st.write("Test Results Shape : ", X_test.shape)
					st.write("Prediction Results : ", y_result.head(5))
					st.write("Line Chart Representation Of Prediction Value:")
					st.line_chart(y_result[:50])

			if regressor == 'Gradient Boosting':

				st.sidebar.subheader("Model Hyper-parameters")
				max_depth = st.sidebar.number_input("The maximum depth of the tree", 5, 20, step=1, key='max_depth')
				min_samples_leaf  = st.sidebar.number_input("The minimum sample leaf", 50, 95, step=5, key='min_samples_leaf')
				min_samples_split  = st.sidebar.number_input("The minimum sample split", 500, 900, step=10, key='min_samples_split')

				#if st.sidebar.button("Classify", key='classify'):
				if st.sidebar.button("Regressor", key='regressor') :
					st.subheader("Gradient Boosting Results")
					model = GradientBoostingRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
					model.fit(X_train,Y_train)
					#accuracy = model.score(X_test, Y_test)
					y_pred = model.predict(X_test)
					mae3 = mean_absolute_error(Y_test, y_pred)
					mse3 = mean_squared_error(Y_test, y_pred)
					y_result = pd.DataFrame(y_pred)
					y_result.rename(columns={0:"Predict"}, inplace=True)
					#d =  {0: '0 - Unassigned', 1: '1 - Low', 2: '2 - Medium', 3: '3 - High'}
					#y_result = y_result["Predict"].map(d)
					#accuracy = model.score(X_test, Y_test)

					plt.figure(figsize=(12,6), dpi=80, facecolor='green')
					#bw = y_result.value_counts()
					#explode=(0.1, 0.1, 0.1, 0.3)
					st.write("Mean Absolute Error : ", mae3)
					st.write("Mean Squared Error : ", mse3)
					#st.write("Confusion Matrix : ", confusion_matrix(Y_test, y_pred))
					st.write("Test Results Shape : ", X_test.shape)
					st.write("Prediction Results : ", y_result.head(5))
					st.write("Line Chart Representation Of Prediction Value:")
					st.line_chart(y_result[:50])

			if regressor == 'Ada Boosting':

				st.sidebar.subheader("Model Hyper-parameters")
				n_estimators = st.sidebar.number_input("n_estimators", 100, 200, step=10, key='n_estimators')
				if st.sidebar.button("Regressor", key='regressor') :
				#if st.sidebar.button("Regressor", key='regressor') :
					st.subheader("Ada Boosting Results")
					model = AdaBoostRegressor(n_estimators=n_estimators)
					model.fit(X_train,Y_train)
					#accuracy = model.score(X_test, Y_test)
					y_pred = model.predict(X_test)
					mae4 = mean_absolute_error(Y_test, y_pred)
					mse4 = mean_squared_error(Y_test, y_pred)
					y_result = pd.DataFrame(y_pred)
					y_result.rename(columns={0:"Predict"}, inplace=True)
					#d =  {0: '0 - Unassigned', 1: '1 - Low', 2: '2 - Medium', 3: '3 - High'}
					#y_result = y_result["Predict"].map(d)
					#accuracy = model.score(X_test, Y_test)

					plt.figure(figsize=(12,6), dpi=80, facecolor='green')

					#st.sidebar.button("Classify", key='classify')
					#bw = y_result.value_counts()

					st.write("Mean Absolute Error : ", mae4)
					st.write("Mean Squared Error : ", mse4)
					#st.write("Confusion Matrix : ", confusion_matrix(Y_test, y_pred))
					st.write("Test Results Shape : ", X_test.shape)
					st.write("Prediction Results : ", y_result.head(5))
					st.write("Line Chart Representation Of Prediction Value:")
					st.line_chart(y_result[:50])


if __name__ == '__main__':
	main()
