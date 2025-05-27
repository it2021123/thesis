from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Συνδυασμός όλων των παραγόμενων αρχείων features
feature_files = [
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/nm_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/koa_El_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/koa_MD_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/koa_SV_features.csv',
    'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/pd_features.csv'
]

# Διαβάζουμε όλα τα features και τα ενώνουμε
dfs = [pd.read_csv(f) for f in feature_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Ορίζουμε τις στήλες που θέλουμε να κρατήσουμε
meta_cols = ['ID','Disease', 'Side', 'Level', 'Window', 'Source_File']
numeric_cols = [
    'Hip_Similarity_Mean','Hip_Similarity_Max','Hip_Similarity_Std',
    'Knee_Similarity_Mean','Knee_Similarity_Max','Knee_Similarity_Std',
    'Ankle_Similarity_Mean','Ankle_Similarity_Max','Ankle_Similarity_Std',
    'Heel_Similarity_Mean','Heel_Similarity_Max','Heel_Similarity_Std',
    'Step_Length_Mean','Step_Length_Max','Step_Length_Std',
    'Left_Hip_Mean','Left_Hip_Max','Left_Hip_Std',
    'Left_Knee_Mean','Left_Knee_Max','Left_Knee_Std',
    'Left_Ankle_Mean','Left_Ankle_Max','Left_Ankle_Std',
    'Left_Heel_Max','Left_Heel_Min',
    'Right_Hip_Mean','Right_Hip_Max','Right_Hip_Std',
    'Right_Knee_Mean','Right_Knee_Max','Right_Knee_Std',
    'Right_Ankle_Mean','Right_Ankle_Max','Right_Ankle_Std',
    'Right_Heel_Max','Right_Heel_Min','Left_Hip_Disp_Mean','Left_Hip_Disp_Max',
    'Left_Hip_Disp_Std','Left_Hip_Vel_Mean','Left_Hip_Vel_Max',
    'Left_Hip_Vel_Std','Left_Knee_Disp_Mean','Left_Knee_Disp_Max','Left_Knee_Disp_Std',
    'Left_Knee_Vel_Mean','Left_Knee_Vel_Max','Left_Knee_Vel_Std','Left_Ankle_Disp_Mean',
    'Left_Ankle_Disp_Max','Left_Ankle_Disp_Std','Left_Ankle_Vel_Mean','Left_Ankle_Vel_Max',
     'Left_Ankle_Vel_Std','Right_Hip_Disp_Mean','Right_Hip_Disp_Max','Right_Hip_Disp_Std',
      'Right_Hip_Vel_Mean','Right_Hip_Vel_Max','Right_Hip_Vel_Std','Right_Knee_Disp_Mean', 
      'Right_Knee_Disp_Max','Right_Knee_Disp_Std','Right_Knee_Vel_Mean','Right_Knee_Vel_Max', 
      'Right_Knee_Vel_Std','Right_Ankle_Disp_Mean','Right_Ankle_Disp_Max','Right_Ankle_Disp_Std', 
      'Right_Ankle_Vel_Mean','Right_Ankle_Vel_Max','Right_Ankle_Vel_Std'

]

# Κρατάμε μόνο τις ζητούμενες στήλες
filtered_df = combined_df[meta_cols + numeric_cols]

# Κανονικοποίηση αριθμητικών δεδομένων με MinMaxScaler
scaler = MinMaxScaler()
filtered_df[numeric_cols] = scaler.fit_transform(filtered_df[numeric_cols])

# Αποθήκευση του τελικού αρχείου
filtered_df.to_csv('C:/Users/giopo/OneDrive/Έγγραφα/thesis/clasifier/all_features.csv', index=False)
print("✅ All features combined and saved to all_features.csv")