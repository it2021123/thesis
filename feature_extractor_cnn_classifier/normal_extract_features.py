# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:27:11 2025

@author: Πουλημένος


Κανονικοποήση εξαγομενων δεδομέων από το cnn
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Συνδυασμός όλων των παραγόμενων αρχείων features
feature_files = [
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/nm_cnn_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/koa_EL_cnn_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/koa_MD_cnn_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/koa_SV_cnn_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/pd_cnn_features.csv'
]

# Διαβάζουμε όλα τα features και τα ενώνουμε
dfs = [pd.read_csv(f) for f in feature_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Δημιουργούμε δυναμικά τη λίστα cnn_feature_cols
cnn_feature_cols = []


for i in range(128):
    for stat in ['mean', 'std', 'min', 'max']:
        cnn_feature_cols.append(f'cnn_feat_{i}_{stat}')

# Ορίζουμε τις στήλες που θέλουμε να κρατήσουμε (meta_cols και τα cnn χαρακτηριστικά)
meta_cols = ['ID', 'Disease', 'Side', 'Level', 'Window', 'Source_File']
numeric_cols = cnn_feature_cols  # Τα χαρακτηριστικά του CNN που δημιουργήσαμε

# Κρατάμε μόνο τις ζητούμενες στήλες
filtered_df = combined_df[meta_cols + numeric_cols]

# Κανονικοποίηση αριθμητικών δεδομένων με MinMaxScaler
scaler = MinMaxScaler()
filtered_df[numeric_cols] = scaler.fit_transform(filtered_df[numeric_cols])

# Αποθήκευση του τελικού αρχείου
filtered_df.to_csv('C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/all_features.csv', index=False)
print("All features combined and saved to all_features.csv")
