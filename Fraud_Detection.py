# setting up environment


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve,
                             classification_report, confusion_matrix,
                             average_precision_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import category_encoders as ce
import shap

np.random.seed(42)
print("✅ Environment ready")