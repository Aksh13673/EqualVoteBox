# =================================================================
# Library: Equal Vote Box (EVB)
# Author: Aksheita Ruchir Dholakia
# Version: 1.0.0 | License: MIT 
# Description: Medical Forest Decision Model
# ==========================================================================================================================================

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

class EqualVoteBox:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.explainer = None
        self.feature_names = None
        self.target_labels = None

    def fit(self, X, y, target_labels=None):
        self.feature_names = X.columns
        self.target_labels = target_labels
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        print("EVB: Model trained and reasoning engine initialized.")

    def show_dashboard(self, patient_data):
   
        all_probs = self.model.predict_proba(patient_data)
        shap_results = self.explainer.shap_values(patient_data)

        for i in range(len(patient_data)):
            print(f"\n--- EVB MEDICAL REPORT (Patient {i+1}) ---")
            print(f"{'Option':<12} | {'Votes':<10} | {'Primary Driver'}")
            print("-" * 55)

            current_probs = all_probs[i]
            for drug_idx, prob in enumerate(current_probs):
                if prob > 0:
                    votes = int(prob * 100)
                    
                    # Smart indexing for SHAP compatibility
                    if isinstance(shap_results, list):
                        patient_drug_impacts = shap_results[drug_idx][i]
                    else:
                        patient_drug_impacts = shap_results[i, :, drug_idx]
                    
                    top_feature_idx = np.argmax(patient_drug_impacts)
                    reason = self.feature_names[top_feature_idx]
                    name = self.target_labels[drug_idx] if self.target_labels is not None else self.model.classes_[drug_idx]
                    
                    print(f"{name:<12} | {votes:>3} / 100 | Driven by {reason}")
        print("\n" + "="*55)
