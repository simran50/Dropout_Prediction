# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Data Analysis
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

#Classifier
from keras.layers import Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score,classification_report

import warnings

warnings.filterwarnings('ignore')

# Load data

train_df = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
train_df.head()

train_df.info()

train_df.columns

# Data analysis

train_df['Target'].unique()

train_df.groupby('Target')['Target'].count()

train_df.describe().T

fig, axes = plt.subplots(3, 2, sharex=True, figsize=(15,10))

sns.kdeplot(data=train_df ,x='Curricular units 2nd sem (approved)', hue='Target', ax=axes[0,0]);
axes[0,0].set_title('Curricular units 2nd sem (approved)');

sns.kdeplot(data=train_df ,x='Curricular units 2nd sem (grade)', hue='Target', ax=axes[0,1]);
axes[0,1].set_title('Curricular units 2nd sem (grade)');

sns.kdeplot(data=train_df ,x='Curricular units 1st sem (approved)', hue='Target', ax=axes[1,0]);
axes[1,0].set_title('Curricular units 1st sem (approved)');

sns.kdeplot(data=train_df ,x='Curricular units 1st sem (grade)', hue='Target', ax=axes[1,1]);
axes[1,1].set_title('Curricular units 1st sem (grade)');

sns.kdeplot(data=train_df ,x='Curricular units 2nd sem (evaluations)', hue='Target', ax=axes[2,0]);
axes[2,0].set_title('Curricular units 2nd sem (evaluations)');

sns.kdeplot(data=train_df ,x='Curricular units 1st sem (evaluations)', hue='Target', ax=axes[2,1]);
axes[2,1].set_title('Curricular units 1st sem (evaluations)');


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10,5))

sns.scatterplot(data=train_df ,x='Curricular units 1st sem (approved)', y='Curricular units 2nd sem (approved)', hue='Target', ax=axes[0]);

sns.scatterplot(data=train_df ,x='Curricular units 1st sem (grade)', y='Curricular units 2nd sem (grade)', hue='Target', ax=axes[1]);

sns.scatterplot(data=train_df ,x='Curricular units 1st sem (evaluations)', y='Curricular units 2nd sem (evaluations)', hue='Target',ax=axes[2]);

def check_and_remove_outliers(df):

    outliers_columns = []
    total_rows = len(df)
    
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outliers.sum()
        outlier_percentage = (outlier_count / total_rows) * 100
        
        if outlier_percentage >= 20:
            outliers_columns.append(column)
            print(f"Outliers detected in column '{column}': {outlier_percentage:.2f}% of total rows.")
            
            # Remove outliers from the DataFrame
            df = df[~outliers]
            print(f"Removed {outlier_count} outliers from column '{column}'.")
    
    if not outliers_columns:
        print("No columns with outliers exceeding 20% detected.")

    return df

df = check_and_remove_outliers(train_df)

fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10,5))

sns.scatterplot(data=df ,x='Curricular units 1st sem (approved)', y='Curricular units 2nd sem (approved)', hue='Target', ax=axes[0]);

sns.scatterplot(data=df ,x='Curricular units 1st sem (grade)', y='Curricular units 2nd sem (grade)', hue='Target', ax=axes[1]);

sns.scatterplot(data=df ,x='Curricular units 1st sem (evaluations)', y='Curricular units 2nd sem (evaluations)', hue='Target',ax=axes[2]);


# data preparation

X=df.drop('Target',axis=1)
y=df['Target'].map({'Graduate':0, 'Dropout':1, 'Enrolled':2})

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

from sklearn.feature_selection import mutual_info_classif

threshold = 20 
high_score_features = []

feature_scores = mutual_info_classif(X, y, random_state=0)

for score, f_name in sorted(zip(feature_scores, X.columns), reverse=True)[:threshold]:
        print(f_name, score)
        high_score_features.append(f_name)
X_new = X[high_score_features]

print(X_new.columns)

