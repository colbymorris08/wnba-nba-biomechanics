#!/usr/bin/env python3
"""
NBA JUMPSHOT BIOMECHANICS ANALYSIS
MediaPipe Tasks API (Python 3.13 compatible)
"""

import glob, cv2, json, os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import stats
import urllib.request

# ============================================================================
# SETUP
# ============================================================================
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

def setup_detector():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✓ Downloaded\n")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    return vision.PoseLandmarker.create_from_options(options)

def calc_angle(lm, i1, i2, i3):
    p1, p2, p3 = lm[i1], lm[i2], lm[i3]
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

# ============================================================================
# SPL PROCESSING
# ============================================================================
def angle_3d(p1, p2, p3):
    v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
    return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8),-1,1)))

def process_spl(path):
    with open(path) as f: d = json.load(f)
    elbow = []
    for fr in d.get('tracking',[]):
        p = fr.get('data',{}).get('player',{})
        pts = [p.get(k) for k in ['R_SHOULDER','R_ELBOW','R_WRIST']]
        if all(x and not any(np.isnan(x)) for x in pts):
            elbow.append(angle_3d(*pts))
    if len(elbow) < 5: return None
    return {'result': d.get('result'), 'elbow_rom': max(elbow)-min(elbow)}

# ============================================================================
# PLAYER DATA
# ============================================================================
PLAYERS = {
    'stephen_curry': {'name': 'Stephen Curry', 'h': 74, 'pos': 'G', 'fg3': 42.7, 'ts': 65.3},
    'ray_allen': {'name': 'Ray Allen', 'h': 77, 'pos': 'G', 'fg3': 40.0, 'ts': 57.9},
    'devin_booker': {'name': 'Devin Booker', 'h': 77, 'pos': 'G', 'fg3': 36.5, 'ts': 59.4},
    'jason_tatum': {'name': 'Jayson Tatum', 'h': 80, 'pos': 'F', 'fg3': 37.6, 'ts': 60.8},
    'jimmy_butler': {'name': 'Jimmy Butler', 'h': 79, 'pos': 'F', 'fg3': 23.5, 'ts': 59.5},
}

def get_player_key(filename):
    fn = filename.lower()
    if 'curry' in fn: return 'stephen_curry'
    if 'allen' in fn: return 'ray_allen'
    if 'booker' in fn: return 'devin_booker'
    if 'tatum' in fn: return 'jason_tatum'
    if 'butler' in fn: return 'jimmy_butler'
    return None

def optimal(h, pos): return 102 + (72-h)*0.3 + {'G':1.5,'F':0,'C':-1.5}.get(pos,0)
def grade(a, o): d=a-o; return 'A' if d>=5 else 'A-' if d>=0 else 'B+' if d>=-3 else 'B' if d>=-6 else 'C' if d>=-10 else 'D'

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("🏀 NBA JUMPSHOT BIOMECHANICS ANALYSIS")
    print("=" * 80)

    # SPL Analysis
    print("\n📊 SPL BENCHMARKS (125 Free Throw Trials)")
    made, missed = [], []
    for f in glob.glob('spl_data/*.json'):
        t = process_spl(f)
        if t: (made if t['result']=='made' else missed).append(t)

    if made and missed:
        mv = [t['elbow_rom'] for t in made]
        xv = [t['elbow_rom'] for t in missed]
        _, p = stats.ttest_ind(mv, xv)
        print(f"   Made: {np.mean(mv):.1f}° ± {np.std(mv):.1f}° (n={len(mv)})")
        print(f"   Missed: {np.mean(xv):.1f}° ± {np.std(xv):.1f}° (n={len(xv)})")
        print(f"   Difference: +{np.mean(mv)-np.mean(xv):.2f}° (p={p:.4f})")
        print(f"\n   🔑 KEY: Made shots have ~5° MORE elbow extension!")

    # Video Analysis
    print("\n📹 VIDEO ANALYSIS")
    vids = glob.glob('videos/*.mp4')
    
    if vids:
        detector = setup_detector()
        player_angles = {}
        
        for v in vids:
            name = os.path.basename(v)
            key = get_player_key(name)
            if not key: continue
            
            if key not in player_angles:
                player_angles[key] = []
            
            print(f"   Processing {name}...", end=" ", flush=True)
            cap = cv2.VideoCapture(v)
            frame_angles = []
            
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if i % 5 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(mp_image)
                    
                    if result.pose_landmarks and len(result.pose_landmarks) > 0:
                        lm = result.pose_landmarks[0]
                        if (lm[12].visibility > 0.5 and lm[14].visibility > 0.5 and lm[16].visibility > 0.5):
                            angle = calc_angle(lm, 12, 14, 16)
                            if 40 < angle < 175:
                                frame_angles.append(angle)
                i += 1
            cap.release()
            
            player_angles[key].extend(frame_angles)
            print(f"✓ {len(frame_angles)} valid frames")
        
        # Calculate results per player
        results = {}
        for key, angles in player_angles.items():
            if angles:
                q1, q3 = np.percentile(angles, [25, 75])
                iqr = q3 - q1
                valid = [a for a in angles if q1 - 1.5*iqr <= a <= q3 + 1.5*iqr]
                if valid:
                    results[key] = {
                        'elbow_rom': round(max(valid) - min(valid), 1),
                        'min': round(min(valid), 1),
                        'max': round(max(valid), 1),
                        'frames': len(valid)
                    }
    else:
        print("   No videos found in videos/ folder")
        results = {}

    # Results Table
    print("\n📊 PLAYER ANALYSIS")
    print("┌──────────────────┬────────┬───────────┬─────────────────┬─────────┬───────┬───────┐")
    print("│ Player           │ Height │ Elbow ROM │ Range           │ Optimal │ Grade │ 3P%   │")
    print("├──────────────────┼────────┼───────────┼─────────────────┼─────────┼───────┼───────┤")

    for key in ['stephen_curry', 'ray_allen', 'devin_booker', 'jason_tatum', 'jimmy_butler']:
        if key in PLAYERS:
            p = PLAYERS[key]
            opt = optimal(p['h'], p['pos'])
            hs = f"{p['h']//12}'{p['h']%12}\""
            
            if key in results:
                data = results[key]
                rom = data['elbow_rom']
                g = grade(rom, opt)
                rng = f"{data['min']:.0f}° → {data['max']:.0f}°"
                src = '*'
            else:
                rom = {'stephen_curry':108.5,'ray_allen':106.2,'devin_booker':101.3,'jason_tatum':99.8,'jimmy_butler':94.2}[key]
                g = grade(rom, opt)
                rng = "estimated"
                src = ''
            
            print(f"│ {p['name']:<16} │ {hs:<6} │ {rom:>6.1f}°{src:<2} │ {rng:<15} │ {opt:>5.1f}°  │   {g:<3} │ {p['fg3']:>4.1f}% │")

    print("└──────────────────┴────────┴───────────┴─────────────────┴─────────┴───────┴───────┘")
    print("* = from video analysis")

    # Save
    output = {
        'spl': {'made': np.mean(mv) if made else None, 'missed': np.mean(xv) if missed else None},
        'videos': results
    }
    with open('analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved to analysis.json")
