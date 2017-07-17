# Day_09_01_10_Classifiers.py
import pandas as pd
import numpy as np
from numpy._distributor_init import NUMPY_MKL  # requires numpy+mkl

from sklearn.preprocessing import LabelEncoder

# from sklearn.metrics import accuracy_score, log_loss
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, NuSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def showInfo(train_file, test_file):
    train_csv = pd.read_csv(train_file)
    # test_csv  = pd.read_csv(test_file)

    print(type(train_csv))
    print(train_csv.shape)

    print(train_csv.species)
    print(train_csv.margin1)
    print('='*50)

    print(train_csv.keys())
    train_csv.info()


showInfo('Leaf/train.csv', 'Leaf/test.csv')









