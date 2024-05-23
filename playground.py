# -*- coding: utf-8 -*-
"""
Created on Sun May  5 01:55:22 2024

@author: mzr0134
"""

from SPanalysis import SpectralAnalysis

dataset_path=r"dataset\hyperspectral\combind.xlsx"

# PLSR
analysis = SpectralAnalysis(dataset_path,'PLSR')
analysis.preprocess_data()
analysis.fit_model()
results = analysis.evaluate_model()
analysis.plot_results()

#SVM
analysis4 = SpectralAnalysis(dataset_path,'SVR')
analysis4.preprocess_data()
analysis4.fit_model()
analysis4.evaluate_model()
analysis4.plot_results()


#RF
analysis5 = SpectralAnalysis(dataset_path,'RF')
analysis5.preprocess_data()
analysis5.fit_model()
analysis5.evaluate_model()
analysis5.plot_results()

#XGboost
XGboost= SpectralAnalysis(dataset_path,'XGBoost')
XGboost.preprocess_data()
XGboost.fit_model()
XGboost.evaluate_model()
XGboost.plot_results()