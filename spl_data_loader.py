"""
SPL Open Data Loader
====================
Fetches and processes the MLSE Sport Performance Lab basketball free throw
biomechanics data from GitHub.

Data source: https://github.com/mlsedigital/SPL-Open-Data
- 125 free throw trials with markerless motion capture
- Joint kinematics time series data
"""

import json
import os
from pathlib import Path
import requests
from typing import Dict, List, Optional, Tuple
import numpy as np

# GitHub API for repository contents
GITHUB_API_BASE = "https://api.github.com/repos/mlsedigital/SPL-Open-Data/contents"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mlsedigital/SPL-Open-Data/main"

class SPLDataLoader:
    """Load and process SPL Open Data basketball biomechanics data."""
    
    def __init__(self, cache_dir: str = "./data/spl_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.participant_info = None
        self.trials = {}
        
    def fetch_file(self, path: str) -> Optional[dict]:
        """Fetch a JSON file from GitHub raw content."""
        url = f"{GITHUB_RAW_BASE}/{path}"
        cache_path = self.cache_dir / path.replace("/", "_")
        
        # Check cache first
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache the data
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            return data
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def list_github_directory(self, path: str) -> List[str]:
        """List files in a GitHub directory via API."""
        url = f"{GITHUB_API_BASE}/{path}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            contents = response.json()
            return [item['name'] for item in contents if item['type'] == 'file']
        except requests.RequestException as e:
            print(f"Error listing {url}: {e}")
            return []
    
    def load_participant_info(self) -> Optional[dict]:
        """Load participant demographic information."""
        self.participant_info = self.fetch_file("basketball/freethrow/participant_information.json")
        return self.participant_info
    
    def discover_trials(self) -> List[str]:
        """Discover available trial files."""
        # Try to list via GitHub API
        # Structure: basketball/freethrow/{participant_id}/trial_data/{trial_id}.json
        participants = self.list_github_directory("basketball/freethrow")
        
        trial_files = []
        for p in participants:
            if p.endswith('.json'):
                continue  # Skip info files
            trials = self.list_github_directory(f"basketball/freethrow/{p}/trial_data")
            for t in trials:
                trial_files.append(f"basketball/freethrow/{p}/trial_data/{t}")
        
        return trial_files
    
    def load_trial(self, trial_path: str) -> Optional[dict]:
        """Load a single trial's motion capture data."""
        data = self.fetch_file(trial_path)
        if data:
            self.trials[trial_path] = data
        return data
    
    def load_all_trials(self, max_trials: int = None) -> Dict[str, dict]:
        """Load all available trials (or up to max_trials)."""
        trial_paths = self.discover_trials()
        
        if max_trials:
            trial_paths = trial_paths[:max_trials]
        
        print(f"Loading {len(trial_paths)} trials...")
        
        for i, path in enumerate(trial_paths):
            self.load_trial(path)
            if (i + 1) % 25 == 0:
                print(f"  Loaded {i + 1}/{len(trial_paths)}")
        
        print(f"Loaded {len(self.trials)} trials successfully")
        return self.trials


def extract_biomechanical_features(trial_data: dict) -> dict:
    """
    Extract key biomechanical features from a trial's motion capture data.
    
    Based on research parameters:
    - Joint angles at prep and release phases
    - Range of motion (ROM) for each joint
    - Angular velocities
    - Release timing
    """
    features = {}
    
    # Expected structure based on SPL data format:
    # trial_data contains time-series joint position/angle data
    
    if 'frames' in trial_data or 'joint_angles' in trial_data:
        # Extract time series
        frames = trial_data.get('frames', trial_data.get('joint_angles', []))
        
        if frames:
            # Key joints for shooting analysis
            joints = ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist']
            
            for joint in joints:
                if joint in frames[0]:
                    angles = [f[joint] for f in frames if joint in f]
                    
                    if angles:
                        # Calculate ROM (max - min)
                        features[f'{joint}_rom'] = max(angles) - min(angles)
                        
                        # Angles at prep (first 20%) and release (last 20%)
                        prep_idx = len(angles) // 5
                        release_idx = -len(angles) // 5 if len(angles) > 5 else -1
                        
                        features[f'{joint}_prep_angle'] = np.mean(angles[:prep_idx]) if prep_idx > 0 else angles[0]
                        features[f'{joint}_release_angle'] = np.mean(angles[release_idx:])
                        
                        # Angular velocity (simplified)
                        if len(angles) > 1:
                            velocities = np.diff(angles)
                            features[f'{joint}_peak_velocity'] = np.max(np.abs(velocities))
                            features[f'{joint}_mean_velocity'] = np.mean(np.abs(velocities))
    
    # Shot outcome if available
    if 'outcome' in trial_data:
        features['made'] = 1 if trial_data['outcome'] in ['made', 'success', True, 1] else 0
    
    # Ball trajectory data if available
    if 'ball' in trial_data:
        ball_data = trial_data['ball']
        if 'release_angle' in ball_data:
            features['ball_release_angle'] = ball_data['release_angle']
        if 'release_height' in ball_data:
            features['ball_release_height'] = ball_data['release_height']
    
    return features


def generate_synthetic_spl_data(n_trials: int = 125, seed: int = 42) -> List[dict]:
    """
    Generate synthetic biomechanics data matching SPL format for testing.
    
    Based on published research values:
    - Cabarkapa et al. (2022, 2023): Free throw biomechanics
    - Frontiers markerless motion capture study
    - Purdue basketball biomechanics research
    
    Research reference values (in degrees):
    - Knee ROM: ~30-50° (prep to extension)
    - Hip ROM: ~20-35°
    - Elbow extension: 130-180° at release
    - Shoulder flexion: 80-120° at release
    - Optimal release angle: ~52°
    """
    np.random.seed(seed)
    
    trials = []
    
    # Research-based parameters (means and std devs)
    params = {
        # Lower body (prep phase flexion -> release extension)
        'hip': {'prep': (140, 8), 'release': (165, 5), 'rom': (25, 5)},
        'knee': {'prep': (130, 10), 'release': (170, 5), 'rom': (40, 8)},
        'ankle': {'prep': (95, 5), 'release': (120, 5), 'rom': (25, 5)},
        
        # Upper body
        'shoulder': {'prep': (60, 10), 'release': (110, 10), 'rom': (50, 10)},
        'elbow': {'prep': (90, 10), 'release': (165, 8), 'rom': (75, 12)},
        'wrist': {'prep': (160, 8), 'release': (200, 10), 'rom': (40, 8)},  # extension past neutral
    }
    
    for i in range(n_trials):
        trial = {
            'trial_id': f'trial_{i+1:03d}',
            'participant_id': 'p001',
            'frames': [],
            'metadata': {
                'fps': 60,
                'duration_frames': np.random.randint(45, 75)  # ~0.75-1.25 seconds
            }
        }
        
        n_frames = trial['metadata']['duration_frames']
        
        # Generate time-series for each joint
        for frame_idx in range(n_frames):
            # Progress through shot (0 = prep, 1 = release)
            t = frame_idx / (n_frames - 1)
            
            frame = {'frame': frame_idx, 'time': frame_idx / 60}
            
            for joint, p in params.items():
                # Smooth transition from prep to release with some noise
                prep_angle = np.random.normal(p['prep'][0], p['prep'][1] * 0.3)
                release_angle = np.random.normal(p['release'][0], p['release'][1] * 0.3)
                
                # Sigmoid-like transition (slower at start/end, faster in middle)
                smooth_t = 1 / (1 + np.exp(-10 * (t - 0.5)))
                
                angle = prep_angle + (release_angle - prep_angle) * smooth_t
                # Add frame-to-frame noise
                angle += np.random.normal(0, 1)
                
                frame[joint] = round(angle, 2)
            
            trial['frames'].append(frame)
        
        # Determine shot outcome based on mechanics
        # Better mechanics (consistent elbow, good knee drive) -> higher make %
        elbow_consistency = 1 - abs(params['elbow']['release'][0] - 165) / 20
        knee_drive = (params['knee']['rom'][0] - 30) / 20
        
        # Base FT% around 75% with mechanical modifiers
        make_prob = 0.75 + 0.1 * elbow_consistency + 0.05 * knee_drive
        make_prob = np.clip(make_prob + np.random.normal(0, 0.1), 0.5, 0.95)
        
        trial['outcome'] = 'made' if np.random.random() < make_prob else 'missed'
        
        # Ball trajectory
        release_frame = trial['frames'][-1]
        trial['ball'] = {
            'release_angle': round(np.random.normal(52, 3), 1),  # Research optimal ~52°
            'release_height': round(np.random.normal(7.5, 0.3), 2),  # feet
            'release_velocity': round(np.random.normal(22, 1.5), 1),  # ft/s
            'entry_angle': round(np.random.normal(45, 4), 1),
        }
        
        trials.append(trial)
    
    return trials


def create_features_dataframe(trials: List[dict]):
    """Convert trials to a pandas DataFrame of extracted features."""
    import pandas as pd
    
    features_list = []
    
    for trial in trials:
        features = extract_biomechanical_features(trial)
        features['trial_id'] = trial.get('trial_id', 'unknown')
        features['outcome'] = trial.get('outcome', 'unknown')
        features_list.append(features)
    
    return pd.DataFrame(features_list)


if __name__ == "__main__":
    print("=" * 60)
    print("SPL OPEN DATA LOADER")
    print("Basketball Free Throw Biomechanics")
    print("=" * 60)
    
    # Try to load real data
    loader = SPLDataLoader()
    
    print("\n1. Attempting to load real SPL data from GitHub...")
    participant_info = loader.load_participant_info()
    
    if participant_info:
        print(f"   Participant info loaded: {participant_info}")
        trials = loader.load_all_trials(max_trials=10)
        print(f"   Loaded {len(trials)} trials")
    else:
        print("   Could not fetch real data. Using synthetic dataset.")
        
    print("\n2. Generating synthetic biomechanics data...")
    synthetic_trials = generate_synthetic_spl_data(n_trials=125)
    print(f"   Generated {len(synthetic_trials)} synthetic trials")
    
    print("\n3. Sample trial structure:")
    sample = synthetic_trials[0]
    print(f"   Trial ID: {sample['trial_id']}")
    print(f"   Frames: {len(sample['frames'])}")
    print(f"   Outcome: {sample['outcome']}")
    print(f"   Ball release angle: {sample['ball']['release_angle']}°")
    
    print("\n4. Sample frame data:")
    frame = sample['frames'][len(sample['frames'])//2]
    for joint in ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist']:
        print(f"   {joint}: {frame[joint]}°")
    
    print("\n5. Extracting features...")
    features = extract_biomechanical_features(sample)
    print("   Key features:")
    for k, v in list(features.items())[:8]:
        print(f"   {k}: {v:.2f}" if isinstance(v, float) else f"   {k}: {v}")
    
    # Save synthetic data
    output_path = Path("./data/synthetic_spl_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(synthetic_trials, f, indent=2)
    print(f"\n✓ Saved synthetic data to {output_path}")
