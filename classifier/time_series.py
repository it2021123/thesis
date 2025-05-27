import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from math import atan2, degrees
from pathlib import Path
import re

def calculate_angle_3d(a, b, c):
    """Υπολογισμός γωνίας ανάμεσα σε 3 σημεία στο 3D (a-b-c)"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    ba_norm = ba / np.linalg.norm(ba)
    bc_norm = bc / np.linalg.norm(bc)
    dot_product = np.dot(ba_norm, bc_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return degrees(angle_rad)


def calculate_similarity(left_vals, right_vals):
    """Calculate similarity metrics between bilateral features"""
    if len(left_vals) == 0 or len(right_vals) == 0:
        return {'mean': np.nan, 'max': np.nan, 'std': np.nan}
    
    # Absolute differences frame-by-frame
    diffs = np.abs(np.array(left_vals) - np.array(right_vals))
    
    return {
        'mean': np.nanmean(diffs),
        'max': np.nanmax(diffs),
        'std': np.nanstd(diffs)
    }

def process_window(window, window_num, nm):
    """Επεξεργάζεται ένα χρονικό παράθυρο δεδομένων βάδισης."""
    dt = 1 / 50.0  # 50 FPS
    features = {}
    columns = ['ID', 'Disease', 'Side'] if nm in ['nm', 'pd'] else ['ID', 'Disease', 'Side', 'Level']
    
    for col in columns:
        if col in window.columns:
            features[col] = window[col].iloc[0]
    features['Window'] = window_num

    # Δοχεία για αρθρώσεις και βήματα
    left_hip, right_hip = [], []
    left_knee, right_knee = [], []
    left_ankle, right_ankle = [], []
    left_heel, right_heel = [], []
    step_lengths = []


    # Νέα: Εκτοπίσεις και ταχύτητες
    joints = ['HIP', 'KNEE', 'ANKLE']
    displacements = {f'{side}_{joint}': [] for side in ['LEFT', 'RIGHT'] for joint in joints}
    velocities = {f'{side}_{joint}': [] for side in ['LEFT', 'RIGHT'] for joint in joints}
    prev_coords = {}

    for _, row in window.iterrows():
        # Αριστερή πλευρά
        left_hip.append(calculate_angle_3d(
            [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y'],row['LEFT_SHOULDER_z']],
            [row['LEFT_HIP_x'], row['LEFT_HIP_y'],row['LEFT_HIP_z']],
            [row['LEFT_KNEE_x'], row['LEFT_KNEE_y'], row['LEFT_KNEE_z']]
        ))
        left_knee.append(calculate_angle_3d(
            [row['LEFT_HIP_x'], row['LEFT_HIP_y'],row['LEFT_HIP_z']],
            [row['LEFT_KNEE_x'], row['LEFT_KNEE_y'], row['LEFT_KNEE_z']],
            [row['LEFT_ANKLE_x'], row['LEFT_ANKLE_y'],row['LEFT_ANKLE_z']]
        ))
       
        left_ankle.append(degrees(atan2(
            row['LEFT_ANKLE_x'] - row['LEFT_KNEE_x'],
            row['LEFT_ANKLE_y'] - row['LEFT_KNEE_y']
        )))
        left_heel.append(row['LEFT_ANKLE_y'])

        # Δεξιά πλευρά
        right_hip.append(calculate_angle_3d(
            [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y'], row['RIGHT_SHOULDER_z']],
            [row['RIGHT_HIP_x'], row['RIGHT_HIP_y'],row['RIGHT_HIP_z']],
            [row['RIGHT_KNEE_x'], row['RIGHT_KNEE_y'], row['RIGHT_KNEE_z']]
        ))
        right_knee.append(calculate_angle_3d(
            [row['RIGHT_HIP_x'], row['RIGHT_HIP_y'],row['RIGHT_HIP_z']],
            [row['RIGHT_KNEE_x'], row['RIGHT_KNEE_y'],row['RIGHT_KNEE_z']],
            [row['RIGHT_ANKLE_x'], row['RIGHT_ANKLE_y'],row['RIGHT_ANKLE_z']]
        ))
    
        right_ankle.append(degrees(atan2(
            row['RIGHT_ANKLE_x'] - row['RIGHT_KNEE_x'],
            row['RIGHT_ANKLE_y'] - row['RIGHT_KNEE_y']
        )))
        right_heel.append(row['RIGHT_ANKLE_y'])

        # Μήκος βήματος
        step_lengths.append(abs(row['LEFT_ANKLE_x'] - row['RIGHT_ANKLE_x']))

        # Εκτοπίσεις & ταχύτητες
        for side in ['LEFT', 'RIGHT']:
            for joint in joints:
                x = row.get(f'{side}_{joint}_x', np.nan)
                y = row.get(f'{side}_{joint}_y', np.nan)
                coord = np.array([x, y])

                key = f'{side}_{joint}'
                if key in prev_coords:
                    disp = np.linalg.norm(coord - prev_coords[key])
                    displacements[key].append(disp)
                    velocities[key].append(disp / dt if dt != 0 else 0)
                else:
                    displacements[key].append(0)
                    velocities[key].append(0)
                prev_coords[key] = coord

    # Ομοιότητες
    for joint, left, right in zip(
        ['Hip', 'Knee', 'Ankle', 'Heel'],
        [left_hip, left_knee, left_ankle, left_heel],
        [right_hip, right_knee, right_ankle, right_heel]
    ):
        sim = calculate_similarity(left, right)
        features[f'{joint}_Similarity_Mean'] = sim['mean']
        features[f'{joint}_Similarity_Max'] = sim['max']
        features[f'{joint}_Similarity_Std'] = sim['std']

    # Στατιστικά μήκους βήματος
    features['Step_Length_Mean'] = np.nanmean(step_lengths)
    features['Step_Length_Max'] = np.nanmax(step_lengths)
    features['Step_Length_Std'] = np.nanstd(step_lengths)

    # Στατιστικά αρθρώσεων ανά πλευρά
    for side, hip, knee, ankle, heel in zip(
        ['Left', 'Right'],
        [left_hip, right_hip],
        [left_knee, right_knee],
        [left_ankle, right_ankle],
        [left_heel, right_heel]
    ):
        for joint, values in zip(['Hip', 'Knee', 'Ankle'], [hip, knee, ankle]):
            features[f'{side}_{joint}_Mean'] = np.nanmean(values)
            features[f'{side}_{joint}_Max'] = np.nanmax(values)
            features[f'{side}_{joint}_Std'] = np.nanstd(values)

        features[f'{side}_Heel_Max'] = np.nanmax(heel)
        features[f'{side}_Heel_Min'] = np.nanmin(heel)

    # Στατιστικά εκτοπίσεων και ταχυτήτων
    for side in ['Left', 'Right']:
        for joint in ['Hip', 'Knee', 'Ankle']:
            key = f'{side.upper()}_{joint.upper()}'
            features[f'{side}_{joint}_Disp_Mean'] = np.nanmean(displacements[key])
            features[f'{side}_{joint}_Disp_Max'] = np.nanmax(displacements[key])
            features[f'{side}_{joint}_Disp_Std'] = np.nanstd(displacements[key])

            features[f'{side}_{joint}_Vel_Mean'] = np.nanmean(velocities[key])
            features[f'{side}_{joint}_Vel_Max'] = np.nanmax(velocities[key])
            features[f'{side}_{joint}_Vel_Std'] = np.nanstd(velocities[key])

    return features

def process_group(input_path, output_path, pattern ,nm):
    """Process all files in a patient group"""
    all_data = []
    
    for csv_file in Path(input_path).glob('*.csv'):
        try:
            # Verify filename pattern
            if not re.search(pattern, csv_file.name):
                continue
                
            df = pd.read_csv(csv_file)
            print(f"Processing {csv_file.name} ({len(df)} frames)")
            
            # Process in 25-frame windows with 12-frame overlap
            window_size = 25
            step_size = 12
            
            for i, window_num in enumerate(range(0, len(df)-window_size+1, step_size)):
                window = df.iloc[window_num:window_num+window_size]
                window_features = process_window(window, i, nm)
                window_features['Source_File'] = csv_file.name
                all_data.append(window_features)
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    # Save results
    if all_data:
        pd.DataFrame(all_data).to_csv(output_path, index=False)

# Configuration for each patient group
groups = [
    {
       'name': 'NM',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/NM/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/nm_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})",
       'nm':'nm'
   },
   {
       'name': 'KOA_EL',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_EL/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/koa_El_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm':'koa'
   },
   {
       'name': 'KOA_MD',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_MD/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/koa_MD_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm':'koa'
   },
   {
       'name': 'KOA_SV',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_SV/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/koa_SV_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm':'koa'
   },
   {
       'name': 'PD',
       'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/PD/',
       'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/classifier/pd_features.csv',
       'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
       'nm':'pd'
       
   }
   
    
]


# Process all groups
for group in groups:
    print(f"\nProcessing {group['name']} group...")
    process_group(group['input'], group['output'], group['pattern'],group['nm'])

print("\nAnalysis complete for all groups!")