# Linear Regression to Predict Titanic Passenger Age


**Deep Dive Coding Data Science Bootcamp Sample Project**

---
Robert Citek - October 5th, 2026



## Problem Definition

State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.  Mention which model will be used, if known.

Statement:

We want to predict the age of a passenger on the titanic given correlated features (e.g., their ticket fare). This is a *supervised regression* data science problem, where the target of the prediction is the age of the passenger.


### Regression

For a regression, the Age will be the target.



## Data Collection/Sources


### Data Overview


This data comes from [Kaggle](https://www.kaggle.com/c/titanic/data), which was then copied to AWS for easy access.

https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv



### Imports

```python
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn import datasets, metrics, model_selection
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
```

### File Paths

Create a variable `url` that contains the web link to the CSV data

```python
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Example/Data/Titanic.train.csv'    # insert url here
url
```

### Load data

Load the data into a data frame named `titanic_df`.

```python
# load data
# titanic_df = pd.read_csv( url, usecols = ['Age', 'Fare'] )
titanic_df = pd.read_csv( url )
titanic_df
```


### head


Look at the first few rows

```python
# first few rows
titanic_df.head()
```


### tail


Look at the last few rows


```python
# last few rows
titanic_df.tail()
```


### shape


```python
# shape of data frame
titanic_df.shape
```

### size


```python
# size of data frame
titanic_df.size
```


### Summary meta-data


Run the `info()` method on the data frame.


```python
# display meta-data
titanic_df.info()
```

### Summary statistics

Summary statistics give you a quick overview of the data.  Some items to spot check:
- the Target:
  - look for nulls
    - null will be culled
  - look at min, max, median, and mean
    - is it a categorical field masquerading as a number?
- the Features:
  - numerical
    - look at min, max, median, and mean
      - a min=1 and max=count suggests an ID
  - categorical
    - look at unique and freq
      - freq=1, unique=count suggests an ID
      - freq=count suggests a meaningless feature


Run the `describe()` method on the data frame.

```python
# display summary statistics
titanic_df.describe()
```


### target nulls



The target should be the first field that you look at.  Its data type will tell you what kind of machine learning problem this is ( classification or regression. ) Any nulls should be culled.  You can check for any skew in the data set.

```python
# define the target column
target_col = 'Age'
```

```python
# look for nulls in the target
titanic_df[target_col].value_counts( dropna = False )
```


```python
# normalized value_counts on target ; include nulls
titanic_df[target_col].value_counts( dropna = False, normalize = True ) * 100
```


### Nulls



```python
# look for nulls
titanic_df.isnull().sum().sort_values( ascending = False )
```


```python
# look for nulls, displaying relative frequency
(titanic_df.isnull().sum() / titanic_df.shape[0]).sort_values( ascending = False ) * 100
```

### dtypes - column data types


Which data types would be appropriate for Linear Regression?


```python
# display data types
titanic_df.dtypes
```

### Unique elements



If the total number of unique items in a field equals the number of rows, that suggests those fields are unique IDs ( e.g. names, IDs ) and should be dropped.


```python
# number of unique elements in a column
titanic_df.nunique().sort_values( ascending=False )
```




## Data Cleaning



### Backup #1


```python
# make backup
titanic_df_bak = titanic_df.copy()
titanic_df_bak
```

```python
# restore from backup
titanic_df = titanic_df_bak.copy()
titanic_df
```


### Keep only the target and one feature


Select `Age` for the target and `Fare` for the feature

```python
# select the target and one feature
target_col = 'Age'
feature_col = 'Fare'
titanic_df = titanic_df.drop( columns = [x for x in titanic_df.columns if x not in [target_col, feature_col]] )
titanic_df
```

### Backup #2


```python
# make backup
titanic_df_bak2 = titanic_df.copy()
titanic_df_bak2
```

```python
# restore from backup
titanic_df = titanic_df_bak2.copy()
titanic_df
```

### Address any rows with nulls


- If the number of nulls in a column is low, it would be easiest to drop the row.
- If the number of nulls in a column is high, it would be easiest to drop the column.
- If the number of nulls in a column is between the low and high, one should impute values for the nulls.


```python
# look for nulls
titanic_df.isnull().sum().sort_values( ascending = False )
```

```python
# compare to totals
titanic_df.count()
```

```python
# drop rows will null values in target column
titanic_df = titanic_df.dropna()
titanic_df
```

```python
# look for nulls
titanic_df.isnull().sum().sort_values( ascending = False )
```

```python
# compare to totals
titanic_df.count()
```

#### Impute


Impute with the median.


```python
# Calculate the median
# median_age = titanic_df['Age'].median()
# median_age
```

```python
# Assign median to nulls
# titanic_df.fillna( {'Age': median_age}, inplace = True )
# titanic_df
```

```python
# Verify there are no nulls
# titanic_df.isnull().sum().sort_values( ascending = False )
```

```python
# Verify median is about the same as before
# titanic_df['Age'].median()
```

## Exploratory Data Analysis



Linear Regression has two requirements:
1. the features are correlated with the target
1. the features are not correlated with each other


### correlated with the target


```python
# Calculate a correlation matrix for the target
column_correlations = titanic_df.corr()
column_correlations
```

```python
# Display the correlation matrix
plt.figure(figsize = (10, 6))
sns.heatmap(
    column_correlations,
    annot = True,
    vmin = -1,
    vmax = 1,
    #cmap = 'RdYlGn'
    cmap = 'coolwarm',
    square = True,
    center=0
)
plt.title("Correlation Matrix: Age vs. Fare")
```

```python
# Create a data series with features ranked by correlation
correlation_matrix = titanic_df.corr()  #.round(2)
correlation_series = correlation_matrix[target_col].abs().sort_values(ascending=False)
correlation_series
```





### independence of features: correlation

```python
# Calculate a correlation matrix for the features
# n/a (only one feature)
```

```python
# Display the correlation matrix
# n/a (only one feature)
```

```python
# Create a data series with features ranked by correlation
# n/a (only one feature)
```

## Processing


### Prepare data frame

```python
target_col = 'Age'
feature_cols = ['Fare']
target_col, feature_cols
```

```python
# Split data frame into target and features
target   = titanic_df[target_col].copy()
features = titanic_df.drop( columns = [target_col] ).copy()
```

```python
# Verify features
features
```

```python
# Verify target
target
```


```python
type(features), features.shape, features.ndim
```

```python
type(target), target.shape, target.ndim
```

### Multiple Cross Validation ( CV )


Create a CV loop with 100 iterations.  Capture performance metric in an array names `results`.
Use accuracy for the performance metric.


```python
# Multiple CV loop
n = 100
accuracies = []
for i in range(0, n):

  # First we will break up our data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)

  # Fit a linear model using Sklearn
  model = LinearRegression()
  myfit = model.fit(X_train, y_train)

  # See how well our model performs on our test data
  y_pred = myfit.predict(X_test)

  # Capture performance metric (accuracy) in array
  accuracies.append(metrics.mean_squared_error(y_test, y_pred)**0.5)

results = np.array(accuracies)
results
```

#### Performance


We'll use RMSE as our performance metric.

```python
# Calculate mean of results
results.mean()
```

```python
# Display a histogram of the results
pd.Series(results).hist(bins = 50)
plt.title("Histogram: RMSE per CV")
plt.xlabel("RMSE")
plt.ylabel("Frequency")
```


## Communication of Results


Conclusion:


Future work:



