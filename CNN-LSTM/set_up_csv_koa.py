"""
Created on Fri Dec  20 11:28:31 2024

@author: Πουλημένος

Καλύτερο format των csv με τις συντεταγμένες αρθρώσεων για να μπορουν
να αναδομηθούν από το Dataset class ωστε να έχοθν την σωστή είσοδο  για 
το CNN-LSTM
"""

import os
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np




# Ορισμός φακέλων
root_folders = [Path("/home/poulimenos/project/output/KOA/"), Path("/home/poulimenos/project/output/NM/"), Path("/home/poulimenos/project/output/PD/")]

# Επιλογή των στηλών που αφορούν τα landmarks (X, Y, Z συντεταγμένες)
landmark_columns = ['LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_z',
                    'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_z',
                    'LEFT_HIP_x', 'LEFT_HIP_y', 'LEFT_HIP_z',
                    'RIGHT_HIP_x', 'RIGHT_HIP_y', 'RIGHT_HIP_z',
                    'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_z',
                    'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_z',
                    'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_ELBOW_z',
                    'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'RIGHT_ELBOW_z',
                    'LEFT_WRIST_x', 'LEFT_WRIST_y', 'LEFT_WRIST_z',
                    'RIGHT_WRIST_x', 'RIGHT_WRIST_y', 'RIGHT_WRIST_z',
                    'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_z',
                    'RIGHT_KNEE_x', 'RIGHT_KNEE_y', 'RIGHT_KNEE_z',
                    'HEAD_x', 'HEAD_y', 'HEAD_z']

# Λίστα με τις αντίστοιχες στήλες αριστερού-δεξιού μέρους
paired_columns = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_ANKLE", "RIGHT_ANKLE"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_KNEE", "RIGHT_KNEE"),
]

count = 0

# Αναζήτηση για .csv αρχεία και ενημέρωση των αντίστοιχων .csv αρχείων
for root_folder in root_folders:
    if not root_folder.exists():
        print(f"Folder not found: {root_folder}")
        continue  # Αν η διαδρομή δεν υπάρχει, παραλείπει τον φάκελο
    csv_files = list(root_folder.rglob("*.csv"))

    for csv_file in csv_files:
        print(f"Found CSV file: {csv_file}")
        filename = os.path.basename(csv_file)
        
        # Εύρεση ID στο όνομα αρχείου
        match = re.search(r"(\d{3})(\w+)_(\d{2})", filename)

        if match:
            # Ανάλυση του ονόματος του αρχείου για εξαγωγή πληροφοριών  +level για koa,pd
            video_id, disease, side = match.groups()
        else:
            print(f"Invalid filename format for {filename}")
            continue  # Αγνόηση αρχείου αν δεν ταιριάζει το πρότυπο
        
        # Φόρτωση δεδομένων από .csv
        try:
            data = pd.read_csv(csv_file)
            print(f"Loaded data from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
        
        # Έλεγχος αν η στήλη `Disease` υπάρχει στο DataFrame
        if 'Disease' not in data.columns:
            print(f"Column 'Disease' not found in {filename}")
            continue
        
        # Έλεγχος για ελλιπή δεδομένα
        missing_data = data[landmark_columns].isnull().sum()
        if missing_data.any():
            print(f"Missing data found in {filename}:")
            print(missing_data[missing_data > 0])
            continue
        
        df_flipped = data.copy()

        # Ανταλλαγή των x, y, z μεταξύ των ζευγών
        for left, right in paired_columns:
            for axis in ["x", "y", "z"]:
                df_flipped[f"{left}_{axis}"], df_flipped[f"{right}_{axis}"] = (
                    data[f"{right}_{axis}"].copy(),
                    data[f"{left}_{axis}"].copy(),
                )

        # Συγχώνευση αρχικών και νέων δεδομένων
        df_augmented = pd.concat([data, df_flipped], ignore_index=True)

        level = ""
        if df_augmented["Disease"][1] == "NM":
            df_augmented['Id'] = (df_augmented["ID"].astype(str) + df_augmented["Disease"] + "_" + df_augmented["Side"].astype(str))
            df_augmented["disease"] = 0
            df_augmented["Disease_Level"] = 0
        elif df_augmented["Disease"][1] == "KOA":
            df_augmented['Id'] = (df_augmented["ID"].astype(str) + df_augmented["Disease"] + '_' + df_augmented["Level"] + '_' + df_augmented["Side"].astype(str))
            df_augmented["disease"] = 1
            if df_augmented['Level'][1] == "MD":
                df_augmented["Disease_Level"] = 1
            elif df_augmented['Level'][1] == "EL":
                df_augmented["Disease_Level"] = 2
            else:
                df_augmented["Disease_Level"] = 3
        else:
            df_augmented['Id'] = (df_augmented["ID"].astype(str) + df_augmented["Disease"] + '_' + df_augmented["Side"].astype(str))
            df_augmented["disease"] = 2
            df_augmented["Disease_Level"] = 4
            level = df_augmented["Level"][1]

        print(df_augmented['Id'])
        
        # Κανονικοποίηση
        if landmark_columns:
            scaler = StandardScaler()
            df_augmented[landmark_columns] = scaler.fit_transform(df_augmented[landmark_columns])
            print("Normalization applied to landmark coordinates.")

        df_augmented.drop(['Disease', 'ID'], axis=1, inplace=True)

        # Ενημέρωση του ίδιου αρχείου CSV με τις τροποποιήσεις
        try:
            if df_augmented["disease"][1] == 0:
                output_folder = Path("/home/poulimenos/project/out_scaler/NM")
                output_csv = output_folder / f"scaler_{video_id}{disease}_{side}.csv"
            elif df_augmented["disease"][1] == 2:
                output_folder = Path("/home/poulimenos/project/out_scaler/PD")
                output_csv = output_folder / f"scaler_{video_id}{disease}_{df_augmented['Level'][1]}_{side}.csv"
            else:
                output_folder = Path("/home/poulimenos/project/out_scaler/KOA")
                output_csv = output_folder / f"scaler_{video_id}{disease}_{df_augmented['Level'][1]}_{side}.csv"

            # Ελέγχουμε αν υπάρχει φάκελος εξόδου, αν όχι, τον δημιουργούμε
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)
            
            # Αποθήκευση των δεδομένων με κωδικοποίηση UTF-8
            df_augmented.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"Processed {filename}, results saved to {output_csv}")
        except Exception as e:
            print(f"Error saving updated file {csv_file}: {e}")
