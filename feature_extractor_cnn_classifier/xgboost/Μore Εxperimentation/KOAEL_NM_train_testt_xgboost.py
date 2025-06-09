# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 01:16:50 2025

@author: Πουλημένος

Πείραμα : Μέθοδος αξιολόγησης διαχωρισμού σε τρία υποσύνολά δεδομένων
(test set ,train set ,validation set)  με XGBoost αλγοριθμό και
 δεδομένα ,στατιστικά μεγέθη ανα χρονικά παράθυρα για κάθε χαρακτηριστικό 
που παράγει το CNN.

Προβλήματα Ταξινόμησεις :
    1)Υγιούς ,σοβαρό στάδιο Οστεοαθρίτιδας
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Χρήσιμη συνάρτηση ===
def convert_commas_to_periods(df):
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].str.replace(',', '.', regex=False)
    return df

# === Φόρτωση αρχείου ===
df = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/thesis/feature_extractor_cnn_classifier/all_features.csv')

df['Disease'] = df['Disease'].str.strip()  # Αφαιρεί τυχόν κενά
df = df[~df['Disease'].str.contains(r'\+?PD', na=False)]


# === Δημιουργία Id ===
df['Id'] = (
    df['ID'].astype(str) +
    df['Disease'].astype(str) +
    df.get('Level', 'NA').astype(str) 
)
# === Καθαρισμός & Προετοιμασία ===
df = df.drop(columns=['ID'], errors='ignore')
df = df.drop(columns=['Source_File'], errors='ignore')

df = convert_commas_to_periods(df)

# === Δημιουργία πεδίων ετικετών ===

df['Disease_Level'] = df['Disease'] + "_" + df['Level'].astype(str)
df = df[~df['Disease_Level'].str.contains(r'\+?KOA_EL', na=False)]
df = df[~df['Disease_Level'].str.contains(r'\+?KOA_MD', na=False)]


# === Label Encoding ===
le_disease = LabelEncoder()
le_disease_level = LabelEncoder()

df['Disease_encoded'] = le_disease.fit_transform(df['Disease'])
df['Disease_Level_encoded'] = le_disease_level.fit_transform(df['Disease_Level'])

# === Κανονικοποίηση χαρακτηριστικών ===
features = df.drop(columns=['Disease', 'Level', 'Disease_Level', 'Disease_encoded', 'Disease_Level_encoded', 'Id'], errors='ignore')
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# === Τελικά δεδομένα ===
y_disease = df['Disease_encoded']
y_disease_level = df['Disease_Level_encoded']
groups = df['Id']

# === Συνάρτηση για LOSO ===
from sklearn.model_selection import GroupShuffleSplit



def train_random_forest_grouped(x, y, groups, message, class_labels=None, show_confusion=True ,num_class=2):
    """
    Εκπαίδευση Random Forest με train-test split που σέβεται τα group IDs.

    Params:
    - x: DataFrame με χαρακτηριστικά.
    - y: Series με ετικέτες.
    - groups: Ομαδοποιήσεις (π.χ. ασθενείς).
    - message: Μήνυμα για εκτύπωση.
    - class_labels: Προαιρετικά labels για confusion matrix.
    - show_confusion: Αν θα εμφανιστεί confusion matrix.

    Returns:
    - clf: Εκπαιδευμένος ταξινομητής.
    - metrics: Dict μετρικών.
    """

    # Χωρισμός βάσει ομάδων
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



    clf = XGBClassifier(
    objective='multi:softprob',    # multiclass με πιθανότητες
    num_class= num_class,                   # βάλε τον αριθμό των κλάσεων σου
    n_estimators=400,              # περισσότερα δέντρα
    learning_rate=0.6,            # 
    max_depth=3,                   # λίγο πιο "βαθιά" δέντρα
    min_child_weight=1,            # μικρότερο βάρος – αντέχει μικρά σύνολα
    gamma=0.02,                     # όχι πολύ σκληρό pruning
    subsample=0.6,                 # drop δείγματα – ενίσχυση γενίκευσης
    eval_metric='mlogloss'
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    if num_class == 2:
        y_pred = np.argmax(y_pred, axis=1)


    # Μετρικές
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_micro": f1_score(y_test, y_pred, average='micro'),
    }

    # Εκτύπωση
    print(f"\n{message}")
    for k, v in metrics.items():
        print(f"{k.capitalize():<13}: {v:.4f}")

    # Confusion Matrix
    if show_confusion:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {message}")
        plt.tight_layout()
        plt.show()

    return clf, metrics


# === Εκτελέσεις ===



# Πρόβλεψη Disease + Level
train_random_forest_grouped(x, y_disease_level, groups, "XGBoost - KOA_SV VS NM", class_labels=le_disease_level.classes_,num_class=2)

