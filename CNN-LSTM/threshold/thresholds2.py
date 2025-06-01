# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:06:16 2025
@author: giopo
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Συνάρτηση ταξινόμησης με thresholds 
def classify_subject(pred_counts, thresholds, margin=0.05):
    total = sum(pred_counts.values())
    probs = {cls: count / total for cls, count in pred_counts.items()}

    passing = {
        cls: probs[cls] - thresholds[cls]
        for cls in probs if probs[cls] >= thresholds[cls]
    }

    if passing:
        sorted_classes = sorted(passing.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_classes) > 1 and (sorted_classes[0][1] - sorted_classes[1][1]) < margin:
            cls1 = sorted_classes[0][0]
            cls2 = sorted_classes[1][0]
            mean_prob1 = probs[cls1]
            mean_prob2 = probs[cls2]
            return cls1 if mean_prob1 >= mean_prob2 else cls2
        else:
            return sorted_classes[0][0]
    else:
        return max(pred_counts, key=pred_counts.get)

# Φόρτωση δεδομένων
df = pd.read_csv('C:/Users/giopo/OneDrive/Έγγραφα/thesis/CNN-LSTM/loso_subject_lstm_levels_results.csv')
y_target = df["Target"]

# Τιμές thresholds για grid search
values = np.arange(0.2, 0.8, 0.1) 


# Αρχικοποίηση μεταβλητών για αποθήκευση των καλύτερων αποτελεσμάτων
best_acc, best_pr, best_rec, best_f1 = 0, 0, 0, 0
list_best_f1 ,list_best_pr ,list_best_rec =0,0,0
best_thresh = {}

# Grid search για όλα τα thresholds
for t0 in values:
    for t1 in values:
        for t2 in values:
            for t3 in values:
                for t4 in values:
                    final_preds = []
                    for _, row in df.iterrows():
                        pred_counts = {
                            0: row["Predicted class 0"],
                            1: row["Predicted class 1"],
                            2: row["Predicted class 2"],
                            3: row["Predicted class 3"],
                            4: row["Predicted class 4"]
                        }
                        thresholds = {0: t0, 1: t1, 2: t2, 3: t3, 4: t4}
                        label = classify_subject(pred_counts, thresholds=thresholds, margin=0.05)
                        final_preds.append(label)

                    # Υπολογισμός ευστοχίας, recall, f1 για κάθε κατηγορία ξεχωριστά
                    precisions = precision_score(y_target, final_preds, average=None, zero_division=0)
                    recalls = recall_score(y_target, final_preds, average=None, zero_division=0)
                    f1s = f1_score(y_target, final_preds, average=None, zero_division=0)

                    # Υπολογισμός ευστοχίας, recall, f1 συνολικά
                    acc = accuracy_score(y_target, final_preds)
                    pr = precision_score(y_target, final_preds, average='macro', zero_division=0)
                    r = recall_score(y_target, final_preds, average='macro', zero_division=0)
                    f1 = f1_score(y_target, final_preds, average='macro', zero_division=0)

                    if f1 > best_f1:
                        best_acc = acc
                        best_f1 = f1
                        best_pr = pr
                        best_rec = r
                        list_best_pr = precisions
                        list_best_re = recalls
                        list_best_f1 = f1s
                        best_thresh = {0: t0, 1: t1, 2: t2, 3: t3, 4: t4}


# Εμφάνιση αποτελεσμάτων
print(f"Grid Search με confidence margin:")
print(f"Καλύτερα thresholds: {best_thresh}")
print(f"Ακρίβεια: {best_acc:.4f}")
print(f"Recall: {best_rec:.4f}")
print(f"Precision: {best_pr:.4f}")
print(f"F1-Macro: {best_f1:.4f}")
for cls in range(5):
    print(f"\nΚατηγορία {cls}:")
    print(f"  Precision: {list_best_pr[cls]:.4f}")
    print(f"  Recall:    {list_best_re[cls]:.4f}")
    print(f"  F1-score:  {list_best_f1[cls]:.4f}")
