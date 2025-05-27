#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:28:31 2025

@author: poulimenos
"""

import os
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np

# Ορισμός φακέλων
root_folders_side_1 = [ Path("/home/poulimenos/project/output/NM/side_1")]
root_folders_side_2=[ Path("/home/poulimenos/project/output/NM/side_2")]

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



def process_data(data, disease, level, filename, landmark_columns=None):
    """
    Επεξεργάζεται τα δεδομένα ανάλογα με την ασθένεια και το επίπεδο της,
    δημιουργεί ένα νέο ID, κανονικοποιεί τις συντεταγμένες των landmarks,
    και αφαιρεί τις στήλες 'Disease' και 'ID'.

    :param data: DataFrame που περιέχει τα δεδομένα
    :param disease: Η ασθένεια (π.χ. "NM", "KOA")
    :param level: Το επίπεδο της ασθένειας (π.χ. "MD", "EL")
    :param filename: Το όνομα του αρχείου για ενημέρωση
    :param landmark_columns: Λίστα με τις στήλες των landmarks για κανονικοποίηση
    :return: Το επεξεργασμένο DataFrame
    """
    if disease == "NM":
        data['Id'] = (data["ID"].astype(str) + data["Disease"] + "_" + data["Side"].astype(str))
        data["disease"] = 0
        data["Disease_Level"] = 0
    elif disease == "KOA":
        data['Id'] = (data["ID"].astype(str) + data["Disease"] + '_' + data["Level"] + '_' + data["Side"].astype(str))
        data["disease"] = 1
        # Χειρισμός των διαφορετικών επιπέδων KOA
        if level == "MD":
            data["Disease_Level"] = 1
        elif level == "EL":
            data["Disease_Level"] = 2
        else:
            data["Disease_Level"] = 3
    else:
        data['Id'] = (data["ID"].astype(str) + data["Disease"] + '_' + data["Side"].astype(str))
        data["disease"] = 2
        data["Disease_Level"] = 4

    print(f"Processed data for {filename}")

    # Κανονικοποίηση των δεδομένων
    if landmark_columns:
        scaler = StandardScaler()
        data[landmark_columns] = scaler.fit_transform(data[landmark_columns])
        print("Normalization applied to landmark coordinates.")

    # Αφαίρεση των στηλών 'Disease' και 'ID'
    data.drop(['Disease', 'ID'], axis=1, inplace=True)

    return data

# Αναζήτηση για .csv αρχεία και ενημέρωση των αντίστοιχων .csv αρχείων
for root_folder1, root_folder2 in zip(root_folders_side_1, root_folders_side_2):
    
    if not root_folder1.exists():
        print(f"Folder not found: {root_folder1}")
        continue  # Αν η διαδρομή δεν υπάρχει, παραλείπει τον φάκελο
    csv_files = list(root_folder1.rglob("*.csv"))
    if not root_folder2.exists():
        print(f"Folder not found: {root_folder2}")
        continue  # Αν η διαδρομή δεν υπάρχει, παραλείπει τον φάκελο
    csv_files2 = list(root_folder2.rglob("*.csv"))

    for csv_file in csv_files:
     print(f"Found CSV file: {csv_file}")
     filename = os.path.basename(csv_file)

    # Ανάλυση του ονόματος του αρχείου
     match = re.search(r"(\d{3})(\w+)_(\d{2})", filename)
     if match:
        video_id, disease,  side = match.groups()
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
     data=process_data(data, disease, "", filename, landmark_columns=landmark_columns)

    # Αναζητούμε τα αντίστοιχα αρχεία για συγχώνευση
     for csv_file2 in csv_files2:
        filename1 = os.path.basename(csv_file2)
        match1 = re.search(r"(\d{3})(\w+)_(\d{2})", filename1)
        if match1:
            video_id2, disease2, side2 = match1.groups()
        else:
            print(f"Invalid filename format for {filename1}")
            continue  # Αγνόηση αν το όνομα του αρχείου δεν ταιριάζει στο πρότυπο
        
        # Συγχώνευση μόνο αν το video_id, disease και level ταιριάζουν
        if video_id == video_id2 and disease == disease2 :
            try:
                data2 = pd.read_csv(csv_file2)
                print(f"Loaded data from {csv_file2}")
                data2=process_data(data2, disease2, "", filename1, landmark_columns=landmark_columns)
                data = pd.concat([data, data2], ignore_index=True)
                break  # Βγαίνουμε από το loop μόλις γίνει η συγχώνευση
            except Exception as e:
                print(f"Error loading {csv_file2}: {e}")
                continue

    # Εξαγωγή του επεξεργασμένου αρχείου
    
     try:
        output_folder = Path("/home/poulimenos/project/out_scaler/NM")
        output_csv = output_folder / f"scaler_{video_id}{disease}_{side}.csv"
       
        
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
  
        data.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Processed {filename}, results saved to {output_csv}")
     except Exception as e:
        print(f"Error saving updated file {csv_file}: {e}")
        
   
