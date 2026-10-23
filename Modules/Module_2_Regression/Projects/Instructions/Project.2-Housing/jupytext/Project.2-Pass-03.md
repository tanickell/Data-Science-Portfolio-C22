# Project 2: Housing Price Prediction – Pass 03


## Problem Definition



State the business problem. Translate the business problem into a Data Science problem by stating what kind of problem it is ( supervised vs unsupervised ) and whether it is a classification, regression, or clustering problem.

Statement:

We want to predict housing sale prices based on various correlated feature data. This is a *supervised regression* problem, where the target of the prediction is sales prices. To accomplish this, we use a regression model alongside Regression prediction techniques, as well as feature scaling and regularization to minimize the RMS percentage error.


## Data Collection/Sources



Data Collection describes the process by which we obtain the data for use in the project. In this case, our data comes from S3 storage buckets on AWS, which we load into memory (variables) via URLs by passing them to the read_csv function of the pandas python library (more on that soon).

The data comes in one large set containing the full data available for use in the project. It is supplemented by a data dictionary that describes the set, the data types of the included data, and other information relevant to its use.

Finally, to store the data in a form we can work with in memory, we use a storage structure called a DataFrame object from the pandas python library. This object allows us to quickly view the data and preview operations on the data without changing it, as well as make updates to the data in a highly intuitive way.


#### 2. Load Pandas, Numpy, and Matplotlib modules

```python
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, metrics
```

#### 3. Load data Train.csv from AWS S3.

```python
# url
url = 'https://ddc-datascience.s3.amazonaws.com/Projects/Project.2-Housing/Data/Housing.Data.csv'
```

- Read in the data from .csv using pandas:

```python
# make dataframe
df = pd.read_csv( url )
```

- Check the dataframe:

```python
# check dataframe
df
```

## Data Cleaning



Data Cleaning is the process of preparing the data set for use in analysis. Many data sets contain bad or null (empty) values, which can cause errors when trying to work with the data and use it in code. Cleaning the data involves removing bad (null-containing) columns and/or rows from the set, as well as imputing (replacing via insertion) similar data into fields with missing or unusable values.

Here, to clean the data, we mostly use the pandas drop function included with pandas DataFrame objects, which allow us to drop columns and rows according to conditions we specify, much like SQL WHERE clauses.


#### 4. Examine the data using tools we have used in class. You should be identifying the various columns such as ...

    - target column
    - feature columns
    - identifier columns
    - null columns
    - null rows

    Tools you should be using include, but are not limited to:

    - head
    - tail
    - shape
    - size
    - info
    - describe
    - nunique
    - unique
    - isnull/isna




First, define the target column:

```python
# define target_col
target_col = 'SalePrice'
target_col
```

- head

```python
# head
df.head()
```

- tail

```python
# tail
df.tail()
```

- shape

```python
# shape
df.shape
```

- size

```python
# size
df.size
```

- info

```python
# info
df.info()
```

- describe

```python
# describe
df.describe()
```

- nunique

```python
# nunique
df.nunique().sort_values( ascending = False )
```

```python
# normalize to look for ID column(s)
df.nunique().sort_values( ascending = False ) / df.shape[0] * 100
```

- unique

```python
# unique
df[target_col].unique()
```

- isnull

```python
# isnull
df.isnull().sum().sort_values( ascending = False )
```

```python
# isnull frequencies
(df.isnull().sum() / df.shape[0]).sort_values( ascending = False ) * 100
```

```python
# look at unique values in the col-by-col null counts to classify cols
((df.isnull().sum() / df.shape[0]).sort_values( ascending = False ) * 100).unique()
```

Many different null counts exist throughout the data set, indicating that this will not be a straightforward nulls cleanup.


- dtypes

```python
# dtypes
df.dtypes
```

```python
# check to see if anything is not zero -- should return 0 if no nulls
df.isnull().sum().sum()
```

#### Target: Value Counts

```python
# value counts on target
df[target_col].value_counts( dropna = False )
```

```python
# value counts normalized
df[target_col].value_counts( dropna = False, normalize = True ) * 100
```

```python
# target col size/shape
df[[target_col]].shape[0]
```

```python
# sum of nulls in target
int(df[[target_col]].isnull().sum().iloc[0])
```

There are no nulls in the target column, which means no rows will need to be dropped on its account.

```python
# sum of non-nulls in target
df[[target_col]].shape[0] - int(df[[target_col]].isnull().sum().iloc[0])
```

```python
# view number of null again
df.isnull().sum()
```

```python
# display columns with all null values
df.isnull().sum()[df.isnull().sum() == df.shape[0]]
```

```python
# count these
int(df.isnull().sum()[df.isnull().sum() == df.shape[0]].count())
```

```python
# show columns with non-null values
df.isnull().sum()[df.isnull().sum() != df.shape[0]]
```

```python
# count these
int(df.isnull().sum()[df.isnull().sum() != df.shape[0]].count())
```

```python
# show only features
df.isnull().sum()[df.isnull().sum() != df.shape[0]][1:-1]
```

```python
# count these
int(df.isnull().sum()[df.isnull().sum() != df.shape[0]][1:-1].count())
```

```python
# make list of columns with all nulls
list_nulls = list(df.isnull().sum()[df.isnull().sum() == df.shape[0]].keys())
print(list_nulls, end = " ")
```

```python
# count the number of list items (should match count of null cols above)
len(list_nulls)
```

```python
# make list of columns with no nulls
list_nulls = list(df.isnull().sum()[df.isnull().sum() == df.shape[0]].keys())
print(list_nulls, end = " ")
```

```python
# columns with no nulls
df.isnull().sum()[df.isnull().sum() == 0]
```

```python
# columns with one or more nulls
df.isnull().sum()[df.isnull().sum() != 0]
```

Summary:

The dataset contains 2637 rows and 81 columns, making for a total of 213597 entries of data. Many of these fields are null.

Details:

  - target column: 'SalePrice'
  - feature columns: [ 'Lot Frontage' ]
  - identifier columns: 'PID'
  - null columns: 0
  - null rows (target): 0

- non-null rows: ?


#### If there are data cleaning issues, develop recommendations for how to deal with them.  Specifically, handle these common issues:

    - identifer columns
    - target column with null counts
    - feature columns with high null counts
    - feature columns with low null counts
    - feature columns with medium null counts


```python
# head again to double-check
df.head()
```

```python
# 1. 'Unnamed' and 'ID_code are very likely identifier columns, and should be dropped; target and var_0 - var_11 will be kept
# 2. 8314 null target rows should be dropped
# 3. 6 null feature columns should be dropped
# 4. After this, drop rows for any column with low null count
# 5. Finally, impute remaining nulls with column medians
```

1. Remove identifier columns

```python
# make backup
df_bak = df.copy()
df_bak
```

```python
# restore from backup
df = df_bak.copy()
df
```

```python
target_col = 'SalePrice'
feature_cols = [ x for x in df.columns if type(df[x][0]) == np.int64 and x != 'SalePrice' ]
target_col, feature_cols
```

```python
# keep the target and all features
df = df.drop( columns = [ x for x in df.columns if x not in [target_col] + feature_cols ] )
```

```python
# check our dataframe (display transformed df)
df
```

2. Drop null target rows




3. Drop feature columns with all nulls


4. Drop remaining records with null counts (if few)


5.  impute with the median for remaining fields

```python
# backup 2
df_bak2 = df.copy()
df_bak2
```

```python
# restore from backup 2
df = df_bak2.copy()
df
```

```python
# check again how many of each column is null
df.isnull().sum().sort_values( ascending = False )
```

```python
# normalize null counts (;er column)
df.isnull().sum().sort_values( ascending = False ) / df.shape[0] * 100
```

```python
# preview filtering rows for where target is not null
df.loc[df[target_col].isnull() == False]
```

Impute:

```python
# loop through features to bulk impute with respective column medians
for feature_col in feature_cols:

  # Calculate the median
  median = df[feature_col].median()
  print(median)

  # Assign median to nulls
  df.fillna( {feature_col: median}, inplace = True )

  # Verify median is about the same as before
  print(df[feature_col].median())
```

```python
df
```

```python
# check again how many of each column is null
df.isnull().sum().sort_values( ascending = False )
```

```python
# backup 3
df_bak3 = df.copy()
df_bak3
```

```python
# restore from backup
df = df_bak3.copy()
df
```

```python
# check total null counts sum
f"Number of remaining nulls in df: {int(df.isnull().sum().sum())}"
```

## Exploratory Data Analysis


```python
len(df.columns)
```

```python
# Let's plot mpg and all of our predictors
subplots_per_row = 4
total_subplots = len(df.columns)
row_count = total_subplots // subplots_per_row
fig, subplots = plt.subplots(row_count,subplots_per_row, figsize = (20,4*row_count)) # Creating 2x4 subplots because we have 8 columns in our data
column_names = df.columns

subplots = subplots.flatten()[:len(column_names)]  # flatten and slice axs
for n, subplot in enumerate(subplots):             # Loop through axs
  subplot.hist(df[column_names[n]])
  subplot.set_xlabel(column_names[n])
  subplot.set_ylabel('Count')
```

### What do we notice?

- For int columns, some features are positively skewed, others negatively skewed, and a couple are evenly distributed
- Lots of positively skewed features (almost look categorical)
- Some features appear categorical (only a few unique values per feature)
- Perhaps some outliers (see positively skewed point above)

```python
# Let's take a look at a correlation plot
plt.figure(figsize=(10,10))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=False, vmin=-1, vmax=1, square=True) ;


```

```python
# Let's take a look at a correlation plot
plt.figure(figsize=(10,10))
correlation_matrix = df.drop( columns = [target_col] ).corr().abs().round(2)
sns.heatmap(data=correlation_matrix, annot=False, vmin=0, vmax=1, square=True) ;


```

```python
correlation_matrix = df.corr().round(2)
correlation_matrix[target_col].abs().sort_values(ascending=False)
```

```python
correlation_matrix = df.corr().round(2)
correlation_matrix[target_col].abs().sort_values(ascending=False).plot( kind="bar", ylim = [0,1])
```

```python
corrs = df.corr().round(2).unstack().abs()
corrs.reset_index()
```

```python
corrs = corrs[corrs < 1]
corrs.reset_index()
```

```python
corrs.sort_values(ascending = False).reset_index()
```

```python
corrs.sort_values(ascending = False)[::2].reset_index()
```

### What do we notice?

- SalePrice correlated with General living area, first floor sf, year built, and overall quality
- lattice pattern: typically, if a feature doesn't correlate with other features, it also doesn't correlate with the target

```python
df.size
```

```python
# Let's plot a pairplot
sns.pairplot(df) ;


```

### What do we notice?

- Not as many non-categorical scatter plots
- some linear-looking trends, including general living area and overall quality (i.e., they highly correlate with the target, and have linear-looking plot clustering)

```python

```

## Processing



```python

```

## Data Visualization/Communication of Results


```python

```
