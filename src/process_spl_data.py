"""
process_spl_data.py
Process SPL motion capture data and extract biomechanical features.

Usage:
    python src/process_spl_data.py
    
Expects raw SPL JSON files in data/spl_trials/
Outputs processed_trials.json to data/
"""

import json
import numpy as np
from pathlib import Path
import math

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "spl_trials"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def load_trial(filepath):
    """Load a trial JSON, handling NaN values."""
    with open(filepath) as f:
        text = f.read().replace('NaN', 'null')
        return json.loads(text)


def compute_joint_angle(p1, p2, p3):
    """
    Compute angle at p2 given three 3D points.
    
    Args:
        p1, p2, p3: Lists of [x, y, z] coordinates
        
    Returns:
        Angle in degrees at p2
    """
    if None in [p1, p2, p3] or any(v is None for v in p1 + p2 + p3):
        return None
    v1 = np.array([p1[i] - p2[i] for i in range(3)])
    v2 = np.array([p3[i] - p2[i] for i in range(3)])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))


def extract_biomechanics(trial):
    """
    Extract biomechanical features from a single trial.
    
    Computes:
        - Knee ROM (range of motion)
        - Hip ROM
        - Elbow ROM
        - Release height
        - Hip drop (squat depth)
    """
    tracking = trial.get('tracking', [])
    if not tracking:
        return None
    
    # Joint angles over time
    knee_angles = []
    hip_angles = []
    elbow_angles = []
    wrist_heights = []
    hip_heights = []
    
    for frame in tracking:
        player = frame.get('data', {}).get('player', {})
        if not player:
            continue
        
        # Right side (shooting arm typically)
        r_hip = player.get('R_HIP')
        r_knee = player.get('R_KNEE')
        r_ankle = player.get('R_ANKLE')
        r_shoulder = player.get('R_SHOULDER')
        r_elbow = player.get('R_ELBOW')
        r_wrist = player.get('R_WRIST')
        l_hip = player.get('L_HIP')
        
        # Knee angle (hip-knee-ankle)
        if r_hip and r_knee and r_ankle:
            knee_angles.append(compute_joint_angle(r_hip, r_knee, r_ankle))
        
        # Hip angle (shoulder-hip-knee)
        if r_shoulder and r_hip and r_knee:
            hip_angles.append(compute_joint_angle(r_shoulder, r_hip, r_knee))
        
        # Elbow angle (shoulder-elbow-wrist)
        if r_shoulder and r_elbow and r_wrist:
            elbow_angles.append(compute_joint_angle(r_shoulder, r_elbow, r_wrist))
        
        # Wrist height (Z coordinate) for release point
        if r_wrist:
            wrist_heights.append(r_wrist[2])
        
        # Hip height for squat depth
        if r_hip and l_hip:
            hip_heights.append((r_hip[2] + l_hip[2]) / 2)
    
    # Filter out None values
    knee_angles = [a for a in knee_angles if a is not None]
    hip_angles = [a for a in hip_angles if a is not None]
    elbow_angles = [a for a in elbow_angles if a is not None]
    
    if not knee_angles or not hip_angles or not elbow_angles:
        return None
    
    # Calculate key metrics
    features = {
        'trial_id': trial['trial_id'],
        'participant_id': trial['participant_id'],
        'result': trial['result'],
        'entry_angle': trial.get('entry_angle'),
        'landing_x': trial.get('landing_x'),
        'landing_y': trial.get('landing_y'),
        
        # Knee metrics
        'knee_min': min(knee_angles),
        'knee_max': max(knee_angles),
        'knee_rom': max(knee_angles) - min(knee_angles),
        'knee_mean': np.mean(knee_angles),
        
        # Hip metrics
        'hip_min': min(hip_angles),
        'hip_max': max(hip_angles),
        'hip_rom': max(hip_angles) - min(hip_angles),
        'hip_mean': np.mean(hip_angles),
        
        # Elbow metrics
        'elbow_min': min(elbow_angles),
        'elbow_max': max(elbow_angles),
        'elbow_rom': max(elbow_angles) - min(elbow_angles),
        'elbow_mean': np.mean(elbow_angles),
        
        # Release point
        'release_height': max(wrist_heights) if wrist_heights else None,
        
        # Hip drop (squat depth)
        'hip_drop': max(hip_heights) - min(hip_heights) if hip_heights else None,
        
        # Frame count
        'n_frames': len(tracking),
    }
    
    return features


def main():
    """Process all SPL trial files."""
    print("=" * 60)
    print("SPL DATA PROCESSOR")
    print("=" * 60)
    
    # Find all trial files
    trial_files = sorted(DATA_DIR.glob("BB_FT_*.json"))
    print(f"\nFound {len(trial_files)} trial files in {DATA_DIR}")
    
    if not trial_files:
        print("ERROR: No trial files found!")
        print(f"Expected files in: {DATA_DIR}")
        return
    
    # Process each trial
    trials = []
    for filepath in trial_files:
        trial = load_trial(filepath)
        features = extract_biomechanics(trial)
        if features:
            trials.append(features)
            print(f"✓ {features['trial_id']}: {features['result']}, "
                  f"entry_angle={features['entry_angle']:.1f}°, "
                  f"elbow_rom={features['elbow_rom']:.1f}°")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Processed {len(trials)} trials successfully")
    
    made = sum(1 for t in trials if t['result'] == 'made')
    missed = len(trials) - made
    print(f"Made: {made}, Missed: {missed}, FT%: {made/len(trials)*100:.1f}%")
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "processed_trials.json"
    
    output = {
        'trials': trials,
        'summary': {
            'total': len(trials),
            'made': made,
            'missed': missed,
            'ft_pct': made / len(trials) * 100
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved to {output_file}")


if __name__ == "__main__":
    main()
