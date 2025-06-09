# -*- coding: utf-8 -*-
"""
Created on Wed May 7 15:09:52 2025
@author: Πουλημένος

Εξαγωγή χαρακτηριστηκών απο δεδομένα βάδισης με χρήση ενός cnn 
νευρωνικου δικτυόυ . Δημιουργία στατιστικών μεγεθών ανα χρονικό
παράθυρο για κλαθε νέο χαρακτηριστικό που δημιουργείται

"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import torch
import torch.nn as nn

#Δομή cnn για εξαγωγή χαρακτηριστικών
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=39, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, feature_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)
 
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu1(x)
        
        x = self.conv3(x)
        x =self.relu1(x)
    
        
        x = self.global_avg_pool(x).squeeze(-1)
        return x


#Συναρτηση που εξάγει δεδομένα απο το cnn
def extract_features_with_cnn(window, feature_extractor):
    """ eξαγωγή χαρακτηριστικών."""
    data = window.filter(regex='_x|_y|_z').values.T
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        cnn_features = feature_extractor(data).numpy().flatten()

    return {f'cnn_feat_{i}': val for i, val in enumerate(cnn_features)}






def process_window(window, window_num, nm, feature_extractor=None):
    """Επεξεργάζεται ένα χρονικό παράθυρο δεδομένων βάδισης με εξαγωγή χαρακτηριστικών."""
    features = {}
    columns = ['ID', 'Disease', 'Side'] if nm in ['nm', 'pd'] else ['ID', 'Disease', 'Side', 'Level']
    
    # Αντιγραφή βασικών χαρακτηριστικών (ID, Disease, Side, Level)
    for col in columns:
        if col in window.columns:
            features[col] = window[col].iloc[0]
    
    features['Window'] = window_num

    # Φιλτράρισμα των στηλών που τελειώνουν σε _x, _y, _z, _visibility
    filtered_columns = [col for col in window.columns if re.search(r'(_x|_y|_z|_visibility)$', col)]
    filtered_data = window[filtered_columns].values
    
    # Εξαγωγή χαρακτηριστικών από CNN αν υπάρχει
    if feature_extractor is not None:
        cnn_features = extract_features_with_cnn(window, feature_extractor)
        cnn_values = list(cnn_features.values())
        
        # Υπολογισμός στατιστικών για τα χαρακτηριστικά CNN
        for i, cnn_feature in enumerate(cnn_values):
            feature_name = f'cnn_feat_{i}'
            features[f'{feature_name}_mean'] = np.mean(cnn_feature)
            features[f'{feature_name}_std'] = np.std(cnn_feature)
            features[f'{feature_name}_min'] = np.min(cnn_feature)
            features[f'{feature_name}_max'] = np.max(cnn_feature)


    # Υπολογισμός στατιστικών για τα φιλτραρισμένα δεδομένα (x, y, z, visibility)
    numeric_data = filtered_data
    features['filtered_mean'] = np.mean(numeric_data)
    features['filtered_std'] = np.std(numeric_data)
    features['filtered_min'] = np.min(numeric_data)
    features['filtered_max'] = np.max(numeric_data)


    return features


#επεξεργασία ανά ομάδα Παθησεις-Κατηγορίας
def process_group(input_path, output_path, pattern, nm, feature_extractor=None):
    all_data = []

    for csv_file in Path(input_path).glob('*.csv'):
        if not re.search(pattern, csv_file.name):
            continue

        df = pd.read_csv(csv_file)
        for i in range(0, len(df) - 25 + 1, 12):
            window = df.iloc[i:i+25]
            features = process_window(window, i, nm, feature_extractor)
            features['Source_File'] = csv_file.name
            all_data.append(features)

    result_df = pd.DataFrame(all_data)

    # Φιλτράρισμα για αριθμητικές στήλες μόνο
    numeric_columns = result_df.select_dtypes(include=[np.number]).columns
    numeric_result_df = result_df[numeric_columns]

    # Υπολογισμός στατιστικών για τις αριθμητικές στήλες
    stats_df = numeric_result_df.describe().T
    stats_df['skew'] = numeric_result_df.skew()
    stats_df['kurtosis'] = numeric_result_df.kurt()

    result_df.to_csv(output_path, index=False)
    stats_df.to_csv(output_path.replace('.csv', '_stats.csv'))



# Αρχικοποίηση του CNN Feature Extractor
cnn_extractor = FeatureExtractor(input_channels=39, feature_dim=64)
cnn_extractor.eval()

# Στοιχεία Λεξικό για την χρήση του σαν ορίσμα  στις παραπάνω συναρτήσεις
groups = [
    {
       'name': 'NM',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/NM/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/nm_cnn_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})",
       'nm': 'nm'
   },
   {
       'name': 'KOA_EL',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_EL/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/koa_El_cnn_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm': 'koa'
   },
   {
       'name': 'KOA_MD',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_MD/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/koa_MD_cnn_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm': 'koa'
   },{
       'name': 'KOA_SV',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_SV/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/koa_SV_cnn_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm': 'koa'
   },
   {
       'name': 'PD',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/PD/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/pd_cnn_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm': 'pd'
   }
]

# Προεπεξεργασία όλων των ομάδων του συνολου δεδομένων
for group in groups:
    print(f"\nProcessing {group['name']} group...")
    process_group(group['input'], group['output'], group['pattern'], group['nm'], cnn_extractor)

print("\nAnalysis complete for all groups!")
