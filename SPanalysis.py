# -*- coding: utf-8 -*-
"""
Created on Sun May  5 01:54:09 2024

@author: mzr0134
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)
from preprocessing import MSC, MeanCenter, Autoscale, trans2absr
from scipy.signal import savgol_filter
from imblearn.combine import SMOTEENN

class SpectralAnalysis:
    '''
    Steps
    
    1. Read the Excel file.
    2. Separate columns for ground truths and reflectance data.
    3.Preprocess the reflectance data using:
        a. Log transformation
        b. Mean scatter correction
        c. Mean-center scaling
    4. Split the dataset into training and testing sets with a 20% test size.
    5. Standardize each ground truth column using a StandardScaler, fitting on the training set and transforming both training and test sets.
    6. Build a PLSR model:
        a. Find the optimum number of components
        b. Train the model using 5-fold cross-validation
        c.Store RMSE and R^2 values for each target during training and cross-validation
    7. Prediction function that computes RMSE, R^2, and prediction values.
    8. Visualization function to plot true and predicted y values from cross-validation and predictions, showing RMSE and R^2.
    
    '''
    
    def __init__(self, filepath: str(), model_type: str() ):
        self.model_type = model_type
        self.data = pd.read_excel(filepath)
        self.models = {}
        self.cv_results = {}
        # parameters
        self.split_ratio = 0.3
        self.random_state=42
        self.cv= 10
    
   
    
    def preprocess_data(self):
        
        self.ground_truths = self.data.loc[:,['RWC','Emm','gsw','PhiPS2','ETR']] #ground truth
        self.reflectance = self.data.iloc[:, 59:1000] #X truth
        self.classes=self.data.loc[:,'Treatment']
 
        
        
        
        #different approach
        
#         reflectance_log = np.log(self.reflectance + 1)
#         reflectance_mean = reflectance_log.mean(axis=1)
#         reflectance_msc = reflectance_log.divide(reflectance_mean, axis=0)
#         self.reflectance_centered = reflectance_msc - reflectance_msc.mean()


        #previous approach
        #reflectance_log = np.log(1/self.reflectance) #log(1/r) transformation\
        
        #reflectance_msc = self.msc(reflectance_log)
        #self.reflectance_centered  = self.mean_center(reflectance_msc)
        
        #filtered_data = savgol_filter(reflectance_log , 17, polyorder = 2,deriv=2)
        #self.reflectance_centered = pd.DataFrame(filtered_data, columns=self.reflectance.columns)
        
        
        
        
        X_train, X_test, y_train_class,y_test_class,y_train_reg, y_test_reg = train_test_split(
            self.reflectance, self.classes,self.ground_truths, test_size=self.split_ratio, random_state= self.random_state
        )
        
        #upsampling and augmentation
        
        X_train, self.y_train_class, self.y_train_reg,self.augmented_data=self.upsample_and_augment(X_train, y_train_class, y_train_reg)

        X_test, self.y_test_class, self.y_test_reg,_=self.upsample_and_augment(X_test, y_test_class, y_test_reg)

        Xp=trans2absr(X_train)
        # Xp=savgol_filter(Xp, 21, 4,2)
        # Xp=pd.DataFrame(Xp)
        # print(type(Xp))
        
        msc = MSC()
   
        Xp = msc.Calibrate(Xp)
        mc= MeanCenter()
        self.X_train = mc.Calibrate(Xp)
                
        auto=Autoscale()
        #self.y_train=auto.Calibrate(y_train)
        
        #Apply the Pre-processing to  the test X data
        Zp=trans2absr(X_test)
        Zp = msc.Apply(Zp)
        self.X_test= mc.Apply(Zp)
        
        #Apply the Pre-processing to  the test X data
        #self.y_test=auto.Apply( y_test )
                
        
        self.scalers = {}
        for column in self.ground_truths.columns:
            scaler = StandardScaler()
            self.y_train_reg[column] = scaler.fit_transform(self.y_train_reg[[column]])
            self.y_test_reg[column] = scaler.transform(self.y_test_reg[[column]])
            self.scalers[column] = scaler

    def upsample_and_augment(self, X, y_class, y_reg, offset_percent=0.06, slope_variation=0.05, num_augments=10):

        df_concat= pd.concat([y_class,y_reg,X],axis=1)
        df_columns_name=df_concat.columns
        print(f"df_columns_name = {df_columns_name.shape}")
        
        df_concat['RWC_category'] = df_concat['RWC'].apply(lambda x: 0 if x > 62 else 1) # 63 is the boundary points
        df_concat['Treatment'] = df_concat['Treatment'].apply(lambda x: 10 if x=="Well watered" else (20 if x=="Mild stress" else 30)) #changing the treatment into numeric value
        
        # extracting X and Y value
        X = df_concat.iloc[:, :-1].values 
        y = df_concat['RWC_category'].values   

        # performing SMOTEENN that combines both oversampling and cleaning oversampled data
        smoteenn = SMOTEENN(sampling_strategy='not majority', random_state=self.random_state)
        X_resampled, y_class_resampled = smoteenn.fit_resample(X, y_class)
        df_concat= df_concat.drop(['RWC_category'],axis=1)
        
        # Calculating the augmentation for the data
        std_dev = X_resampled[:,6:].std()
        # print(std_dev)
        augmented_data = []
        for _ in range(num_augments):
            for _, row in df_concat.iterrows():
                augmented_row = row.copy()
                offset = np.random.uniform(-offset_percent * std_dev, offset_percent * std_dev)
                slope = 1 + np.random.uniform(-slope_variation, slope_variation)
                augmented_features = augmented_row.iloc[:-1].astype(float) * slope + offset
                augmented_data .append(list(augmented_features) + [augmented_row.iloc[-1]])
    
        augmented_data=pd.DataFrame(augmented_data,columns=df_concat.columns) #converting list to df
        augmented_data["Treatment"]=augmented_data["Treatment"].apply(lambda x: "Well watered" if 0<x<15 else ("Mild stress" if 18<x<23 else "Extreme stress"))

        # Separating the samples
        y_reg_augmented = augmented_data.loc[:,['RWC','Emm','gsw','PhiPS2','ETR']]
        y_class_augmented =augmented_data.loc[:,['Treatment']]
        X_augmented = augmented_data.iloc[:,6:]
        
        return X_augmented, y_class_augmented, y_reg_augmented,augmented_data
    
    def fit_model(self):
        print("Model type:", self.model_type)
        if self.model_type == 'PLSR':
            self.Model_PLSR()
        elif self.model_type == 'SVR':
            self.model_svr()
        elif self.model_type == 'RF':
           
            self.model_random_forest()
        elif self.model_type == 'XGBoost':
            self.model_xgboost()
        elif self.model_type == 'MLP':
            # Placeholder for MLP
            pass
        else: 
            print('No model selected')

    def Model_PLSR(self, max_components=6):
        print('Here')
        for idx, target in enumerate(self.y_train_reg.columns):
            min_rmse = float('inf')
            best_n_components = 1
            for n_components in range(1, min(max_components, self.X_train.shape[1]) + 1):
                pls = PLSRegression(n_components=n_components)
                y_cv = cross_val_predict(pls, self.X_train, self.y_train_reg[target], cv=self.cv)
                rmse = np.sqrt(mean_squared_error(self.y_train_reg[target], y_cv))
                r2 = r2_score(self.y_train_reg[target], y_cv)
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_n_components = n_components
                    self.cv_results[target] = {'RMSE': rmse, 'R2': r2}
            best_pls = PLSRegression(n_components=best_n_components)
            best_pls.fit(self.X_train, self.y_train_reg[target])
            self.models[target] = best_pls
            print(f"Best number of components for {target}: {best_n_components}")

    def model_svr(self):
        
        """
        Support vector machine (regression)
        
        """
#         param_distributions = {
#             'C': [0.01, 0.1, 1, 10, 15, 100],
#             'gamma': [0.1, 0.001, 0.01, 1],
#             'kernel': ['rbf']
#         }
        param_distributions = {
            'C': [0.0001,0.001,0.01,0.1,1,10,11,12,13,14,100,200,300,500,1000,2000,4000],  # Expanding the range of C
            'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto'],  # Adding more options, including 'scale' and 'auto'
            'kernel': ["poly","rbf",'linear'],  # Exploring different kernel types
            'epsilon': [0.01, 0.1, 0.5,0.55,0.6,0.65,0.75,0.7,0.8,0.85],  # Epsilon in the epsilon-SVR model
            'degree': [2,3, 4,5,6,7,10]  # Degree of the polynomial kernel function (if 'poly' is chosen)
        }

        random_search = RandomizedSearchCV(
            estimator=SVR(),
            param_distributions=param_distributions,
            n_iter=100,
            verbose=1,
            random_state=self.random_state,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        for column in self.y_train_reg.columns:
            print(f"Training SVR for {column}")
            random_search.fit(self.X_train, self.y_train_reg[column])
            best_svr = random_search.best_estimator_
            self.models[column] = best_svr
            print(f"Best parameters for {column}: {random_search.best_params_} \n")
            print(f"Best score (MSE) for {column}: {random_search.best_score_}")
    
    def model_random_forest(self):

        """
        Random forest model

        """

        # Define the parameter grid for Random Forest
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum number of levels in tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
            'bootstrap': [True, False]  # Method of selecting samples for training each tree
        }

        # Initialize a RandomizedSearchCV object using RandomForestRegressor as the estimator
        random_search = RandomizedSearchCV(
            estimator=RandomForestRegressor(),
            param_distributions=param_distributions,
            n_iter=10,  # Adjust based on computational budget
            cv=self.cv,  # Number of folds in cross-validation
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )

        # Dictionary to store the models for each target


        # Fit Random Forest model for each target
        for column in self.y_train_reg.columns:
            print(f"Training Random Forest for {column}")
            # Conduct the random search
            random_search.fit(self.X_train, self.y_train_reg[column])

            # Best estimator after the search
            best_rf = random_search.best_estimator_

            # Store the best model in the dictionary
            self.models[column] = best_rf

            # Optionally, print the best parameters and score
            print(f"Best parameters for {column}: {random_search.best_params_}")
            print(f"Best score for {column}: {random_search.best_score_}")
            
            
    def model_xgboost(self):
        # Define the parameter grid for XGBoost


        # Dictionary to store the models for each target

        # Fit XGBoost model for each target
        for column in self.y_train_reg.columns:
            print(f"Training XGBoost for {column}")

            # Store the best model in the dictionary
            model = XGBRegressor(n_estimators=1000, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
            model.fit(self.X_train, self.y_train_reg[column])
            self.models[column] = model
            
    def evaluate_model(self):
        results = {}
        for target, model in self.models.items():
            y_pred = model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test_reg[[target]], y_pred))
            r2 =  model.score(self.X_test,self.y_test_reg[target])  # Ensuring R2 is not negative
            results[target] = {'RMSE': rmse, 'R2': r2}
        return results

    def plot_results(self):
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
        axs = axs.flatten()
        for i, target in enumerate(self.models.keys()):
            model = self.models[target]

            # Predictions
            y_pred_scaled = model.predict(self.X_test)
 
            
            # Check for dimension
            if y_pred_scaled.ndim == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = self.scalers[target].inverse_transform(y_pred_scaled)
            y_true_test = self.scalers[target].inverse_transform(self.y_test_reg[[target]])
            
            # Cross-validation
            y_cv_scaled = cross_val_predict(model, self.X_train, self.y_train_reg[target], cv=self.cv)

            
            # Check for dimension
            if y_cv_scaled.ndim == 1:
                y_cv_scaled = y_cv_scaled.reshape(-1, 1)
                
            y_cv = self.scalers[target].inverse_transform(y_cv_scaled)
            y_true_train = self.scalers[target].inverse_transform(self.y_train_reg[[target]])

            # RMSE and R2 for predictions
            
            rmse_pred = np.sqrt(mean_squared_error(self.y_test_reg[target], y_pred_scaled))
            r2_pred = abs(model.score(self.X_test,self.y_test_reg[target])) # Ensuring R2 is not negative

            # RMSE and R2 for cross-validation

            rmse_cv = np.sqrt(mean_squared_error(self.y_train_reg[target], y_cv_scaled))
            r2_cv = abs(model.score(self.X_train,self.y_train_reg[target])) # Ensuring R2 is not negative

            # Plotting
            axs[i].scatter(y_true_test, y_pred, color='red', label=f'Prediction (RMSE={rmse_pred:.2f}, R²={r2_pred:.2f})')
            axs[i].scatter(y_true_train, y_cv, color='blue', label=f'CV (RMSE={rmse_cv:.2f}, R²={r2_cv:.2f})')

            # Trend lines
            # Calculate coefficients for trend lines
            z_pred = np.polyfit(y_true_test.flatten(), y_pred.flatten(), 1)
            p_pred = np.poly1d(z_pred)
            z_cv = np.polyfit(y_true_train.flatten(), y_cv.flatten(), 1)
            p_cv = np.poly1d(z_cv)

            # Plot trend lines
            axs[i].plot(y_true_test, p_pred(y_true_test), "r--")
            axs[i].plot(y_true_train, p_cv(y_true_train), "b--")

            axs[i].set_title(f'Results for {target}')
            axs[i].set_xlabel('True Values')
            axs[i].set_ylabel('Predicted Values')
            axs[i].legend()

        plt.tight_layout()
        plt.show()
        
    def mean_center(self,value):
        """
        Preprocessing
        """
        for i in range(value.shape[0]):
             value.iloc[i,:]= (value.iloc[i,:]-np.mean(value.iloc[i,:]))
        return value

    def msc(self,value,reference=None):
        """
        Preprocessing
        """
        data_msc= np.zeros_like(value)
        for i in range(value.shape[0]):
            
            fit= np.polyfit(ref, y, deg)

        return (data_msc)