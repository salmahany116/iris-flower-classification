from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

"""# load Data"""

iris_data=datasets.load_iris()
features=iris_data.data
labels=iris_data.target

df = pd.DataFrame(data=iris_data.data,
                  columns=iris_data.feature_names)

# Add the target variable to the dataframe
df['target'] = iris_data.target

"""### External csv file
*. from kaggle :https://www.kaggle.com/datasets/uciml/iris
"""

df.head()

"""# Process the Data"""

#checking for null values
df.isnull().sum()

df.columns

# #Drop unwanted columns
# df=df.drop(columns="Id")

"""# Split the data"""

X_train,X_test,y_train,y_test=train_test_split(features,labels,random_state=0)

"""# Train the model"""

# Create a logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Train the model on the training data
logreg.fit(X_train, y_train)

y_pred=logreg.predict(X_test)

y_pred

y_test

"""# Accuracy"""

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

"""## Test"""

iris1=[[5.1,3.5,1.4,0.2]]

irisPredictLabel=['Setosa','Versicolor','Virginica']

y_pred=logreg.predict(iris1)

irisPredictLabel[y_pred[0]]