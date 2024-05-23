"""
Created on Sat Feb 17 18:51:58 2024

@author: mzr0134
"""

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#loading dataset
dataset=pd.read_excel('dataset/onlyRWC.xlsx')
print(f'Total sample {dataset.shape}')


##helper function
# Categorizing RWC values into specified ranges
def categorize_rwc(rwc):
    print()
    if 100>= rwc > 90:
       
        return 'High'
    elif 90>= rwc > 80:
       
        return 'Mid'
    else:
        
        return 'Low'
def categorize_rwc_two(rwc):
    print()
    if 100>= rwc > 80:
       
        return 'High'
    else:
        return 'Low'
    
def rwcclasstonumeric_two(cls):
    if cls == "High":
        return 0
    else:
        return 1
       
def categorize_rwc_two(rwc):
    print()
    if 100>= rwc > 80:
       
        return 'High'
    else:
        return 'Low'

def numeric_class(cls):
    if cls == "WW":
        return 0
    elif cls == "MS":
        return 1
    else:
        return 2
    
def rwcclasstonumeric(cls):
    if cls == "High":
        return 0
    elif cls == "Mid":
        return 1
    else:
        return 2
    
def invers_numeric_class(cls):
    if cls == 0:
        return "WW"
    elif cls == 1:
        return "MS"
    else:
        return "ES"
    
def SMOTE_fn(dataset):
    X = dataset.iloc[:, 5:-1]  # Selecting columns from 34 to the end (excluding the last two added columns)
    y = dataset['RWC_Category']  # The encoded RWC categories
    X.columns = X.columns.astype(str)
    # Checking the balance of the categories before applying SMOTE
    print(f'class distribution{y.value_counts()}')

    #performing balancing
    smote = SMOTE(random_state=42,k_neighbors=9)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
#creating new column based on rwc range from 100-90, 80-90, and 80-60.
#changing the class columns value to 0,1, and 2. for SMOTE function.
#Changing the class and RWC columns name to 0 and 1 for SMOTE function

dataset['RWC_Category'] = dataset['RWC'].apply(categorize_rwc)
dataset['class'] = dataset['class'].apply(numeric_class)
dataset.rename(columns={'class': 0, 'RWC': 1}, inplace=True)

sns.displot(dataset["1"])
# Encoding the RWC categories and class for use with SMOTE
dataset['RWC_Category']=dataset['RWC_Category'].apply(rwcclasstonumeric)


high_df= dataset[dataset['RWC_Category']==0]
low_df=dataset[dataset['RWC_Category']==2]
medium_df=dataset[dataset['RWC_Category']==1]
medium_df.columns =medium_df.columns.astype(str)
highlow_df=pd.concat([high_df,low_df],axis=0)

sns.distplot(low_df[1])
sns.distplot(medium_df[1])
sns.distplot(high_df[1])
#################################################################################
#manual checking
# Selecting the features (X) and the target (y)
# X = dataset.iloc[:, 5:-1]  # Selecting columns from 34 to the end (excluding the last two added columns)
# y = dataset['RWC_Category']  # The encoded RWC categories
# X.columns = X.columns.astype(str)
# # Checking the balance of the categories before applying SMOTE
# y.value_counts()

# #performing balancing
# smote = SMOTE(random_state=42,k_neighbors=9)
# X_resampled, y_resampled = smote.fit_resample(X, y)
#################################################################################

X_resampled, y_resampled=SMOTE_fn(highlow_df)
X_resampled=pd.concat([X_resampled,medium_df],axis=0)
X_resampled.rename(columns={'0': 'class', "1":'RWC'}, inplace=True)
#distribution after resampling 
sns.distplot(X_resampled["RWC"])

#revert back to orginal name

X_resampled['class']=X_resampled['class'].apply(invers_numeric_class)

# Checking the balance of the categories after applying SMOTE
y_resampled.value_counts()

#checking the current sample distribution
high= X_resampled[(100.00>=X_resampled["RWC"]) & (X_resampled["RWC"]>90.00)]
Medium= X_resampled[(90.00>=X_resampled["RWC"]) & (X_resampled["RWC"]>80.00)]
low= X_resampled[(80.00>=X_resampled["RWC"]) & (X_resampled["RWC"]>50.00)]

print(f'sample in range 100-90% ={high.shape[0]} which is {(high.shape[0])/X_resampled.shape[0] *100}% of the dataset')
print(f'sample in range 90-80% ={Medium.shape[0]} which is {(Medium.shape[0])/X_resampled.shape[0] *100}% of the dataset')
print(f'sample in range 80-60% ={low.shape[0]} which is {(low.shape[0])/X_resampled.shape[0] *100}% of the dataset')

#checking statistics
ans=X_resampled.groupby(['class'])['RWC'].describe()
#checking distribution plot
sns.distplot(X_resampled['RWC'])



#--------------- ONLY TWO CLASS OF RWC 100-80 AND 80-0--------------------------------------

dataset['RWC_Category'] = dataset['RWC'].apply(categorize_rwc_two)
dataset['class'] = dataset['class'].apply(numeric_class)
dataset.rename(columns={'class': 0, 'RWC': 1}, inplace=True)

  
# Encoding the RWC categories and class for use with SMOTE
dataset['RWC_Category']=dataset['RWC_Category'].apply(rwcclasstonumeric_two)


high_df= dataset[dataset['RWC_Category']==0]
low_df=dataset[dataset['RWC_Category']==1]

highlow_df=pd.concat([high_df,low_df],axis=0)

sns.distplot(low_df[1])
sns.distplot(high_df[1])
#################################################################################
#manual checking
# Selecting the features (X) and the target (y)
# X = dataset.iloc[:, 5:-1]  # Selecting columns from 34 to the end (excluding the last two added columns)
# y = dataset['RWC_Category']  # The encoded RWC categories
# X.columns = X.columns.astype(str)
# # Checking the balance of the categories before applying SMOTE
# y.value_counts()

# #performing balancing
# smote = SMOTE(random_state=42,k_neighbors=9)
# X_resampled, y_resampled = smote.fit_resample(X, y)
#################################################################################

X_resampled, y_resampled=SMOTE_fn(dataset)


#distribution after resampling 
sns.distplot(X_resampled["1"])

#revert back to orginal name
X_resampled.rename(columns={'0': 'class', "1":'RWC'}, inplace=True)
X_resampled['class']=X_resampled['class'].apply(invers_numeric_class)

# Checking the balance of the categories after applying SMOTE
y_resampled.value_counts()

#checking the current sample distribution
high= X_resampled[(100.00>=X_resampled["RWC"]) & (X_resampled["RWC"]>90.00)]
Medium= X_resampled[(90.00>=X_resampled["RWC"]) & (X_resampled["RWC"]>80.00)]
low= dataset[(80.00>=X_resampled["RWC"]) & (X_resampled["RWC"]>0]

print(f'sample in range 100-90% ={high.shape[0]} which is {(high.shape[0])/X_resampled.shape[0] *100}% of the dataset')
print(f'sample in range 90-80% ={Medium.shape[0]} which is {(Medium.shape[0])/X_resampled.shape[0] *100}% of the dataset')
print(f'sample in range 80-60% ={low.shape[0]} which is {(low.shape[0])/X_resampled.shape[0] *100}% of the dataset')

#checking statistics
ans=X_resampled.groupby(['class'])['RWC'].describe()
#checking distribution plot
sns.distplot(X_resampled['RWC'])







