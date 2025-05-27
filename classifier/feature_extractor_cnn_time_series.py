# -*- coding: utf-8 -*-
"""
Advanced Gait Analysis Pipeline with Multiple Feature Extraction Methods
"""

f
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
import pywt
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. Enhanced Feature Extractor with Multiple Methods
class GaitFeatureExtractor:
    def __init__(self, methods=['cnn', 'statistical', 'spectral']):
        self.methods = methods
        self.cnn_extractor = self._init_cnn() if 'cnn' in methods else None
        
    def _init_cnn(self):
        """Initialize 1D CNN feature extractor"""
        model = nn.Sequential(
            nn.Conv1d(39, 64, kernel_size=3, padding=1),  # 39 input channels
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        model.eval()
        return model
    
    def extract_features(self, window):
        """Main feature extraction method"""
        features = {}
        
        # Convert to numpy array once
        data = window.filter(regex='_x|_y|_z').values.T  # (39, 25)
        
        if 'cnn' in self.methods and self.cnn_extractor:
            features.update(self._extract_cnn_features(data))
            
        if 'statistical' in self.methods:
            features.update(self._extract_statistical_features(data))
            
        if 'spectral' in self.methods:
            features.update(self._extract_spectral_features(data))
            
        if 'wavelet' in self.methods:
            features.update(self._extract_wavelet_features(data))
            
        return features
    
    def _extract_cnn_features(self, data):
        """Extract deep features using CNN"""
        try:
            with torch.no_grad():
                tensor = torch.FloatTensor(data).unsqueeze(0)  # (1, 39, 25)
                features = self.cnn_extractor(tensor).numpy().flatten()
                return {f'cnn_{i}':v for i,v in enumerate(features)}
        except Exception as e:
            print(f"CNN extraction failed: {e}")
            return {}


    def _extract_statistical_features(self, data):
        """Traditional statistical features"""
        features = {}
        for i, channel in enumerate(data):
            prefix = f'ch{i}_'
            features.update({
                prefix+'mean': np.mean(channel),
                prefix+'std': np.std(channel),
                prefix+'skew': stats.skew(channel),
                prefix+'kurtosis': stats.kurtosis(channel),
                prefix+'rms': np.sqrt(np.mean(channel**2))
            })
        return features

    def _extract_spectral_features(self, data):
        """Frequency domain features"""
        features = {}
        for i, channel in enumerate(data):
            f, Pxx = signal.welch(channel)
            prefix = f'freq_{i}_'
            features.update({
                prefix+'peak_freq': f[np.argmax(Pxx)],
                prefix+'total_power': np.sum(Pxx),
                prefix+'spectral_entropy': stats.entropy(Pxx+1e-12)
            })
        return features

    def _extract_wavelet_features(self, data):
        """Wavelet transform features"""
        features = {}
        wavelet = 'db4'
        for i, channel in enumerate(data):
            coeffs = pywt.wavedec(channel, wavelet, level=3)
            for j, coeff in enumerate(coeffs):
                prefix = f'wave_{i}_l{j}_'
                features.update({
                    prefix+'energy': np.sum(coeff**2),
                    prefix+'std': np.std(coeff)
                })
        return features

# 2. Data Pipeline with Generators
class GaitDataPipeline:
    def __init__(self, config):
        self.config = config
        self.feature_extractor = GaitFeatureExtractor(
            methods=config.get('methods', ['cnn', 'statistical'])
        )
        
    def process_all_groups(self):
        """Process all patient groups"""
        results = {}
        for group_name, group_config in self.config['groups'].items():
            print(f"\nProcessing {group_name} group...")
            group_results = self._process_group(group_config)
            if group_results:
                self._save_results(group_results, group_config['output'])
                results[group_name] = group_results
        return results
    
    def _process_group(self, group_config):
        """Process single patient group"""
        all_features = []
        files = list(Path(group_config['input']).glob('*.csv'))
        
        for csv_file in tqdm(files, desc=f"Processing {group_config['name']}"):
            try:
                for window_features in self._process_file(csv_file, group_config):
                    window_features['source_file'] = csv_file.name
                    all_features.append(window_features)
            except Exception as e:
                print(f"\nError processing {csv_file.name}: {e}")
                continue
                
        return pd.DataFrame(all_features) if all_features else None
    
    def _process_file(self, csv_file, group_config):
        """Generator that yields features for each window"""
        df = pd.read_csv(csv_file)
        window_size = self.config.get('window_size', 25)
        step_size = self.config.get('step_size', 12)
        
        for i in range(0, len(df)-window_size+1, step_size):
            window = df.iloc[i:i+window_size]
            
            # Basic metadata
            features = {
                'ID': window['ID'].iloc[0],
                'Disease': window['Disease'].iloc[0],
                'Side': window['Side'].iloc[0],
                'window_num': i
            }
            
            if 'Level' in window.columns:
                features['Level'] = window['Level'].iloc[0]
            
            # Extract features
            features.update(self.feature_extractor.extract_features(window))
            yield features
    
    def _save_results(self, df, output_path):
        """Save results with post-processing"""
        # Normalize features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path} ({len(df)} samples)")

# 3. Configuration and Main Execution
if __name__ == "__main__":
    # Configuration dictionary
    config = {
        'window_size': 25,
        'step_size': 12,
        'methods': ['cnn', 'statistical', 'spectral'],
        'groups': {
            'NM': {
                'name': 'Normal Controls',
                'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/NM/',
                'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/features/nm_features.csv'
            },
            'KOA_EL': {
                'name': 'KOA Early Stage',
                'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_EL/',
                'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/features/koa_el_features.csv'
            },
            'KOA_MD': {
                'name': 'KOA Moderate Stage',
                'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/KOA/KOA_MD/',
                'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/features/koa_md_features.csv'
            },
            'PD': {
                'name': 'Parkinson\'s Disease',
                'input': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/output/PD/',
                'output': 'C:/Users/giopo/OneDrive/Έγγραφα/thesis/features/pd_features.csv'
            }
        }
    }

    # Execute pipeline
    pipeline = GaitDataPipeline(config)
    results = pipeline.process_all_groups()
    
    print("\nFeature extraction completed successfully!")