"""
Streamlit App: Chess-Style Play Analysis
========================================
Interactive evaluation of Phoenix Suns plays for NBA vs WNBA.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Page config
st.set_page_config(
    page_title="Basketball Play Evaluator",
    page_icon="♟️",
    layout="wide"
)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayBranch:
    name: str
    shot_zone: str
    success_prob: float
    description: str
    player_positions: Dict[int, Tuple[float, float]]  # End positions after action

@dataclass
class Play:
    name: str
    category: str
    initial_positions: Dict[int, Tuple[float, float]]
    branches: List[PlayBranch]
    description: str


# =============================================================================
# ZONE EFFICIENCY DATA
# =============================================================================

DEFAULT_NBA_EFFICIENCY = {
    'rim': 1.30,
    'paint_non_ra': 0.82,
    'midrange': 0.80,
    'corner_3': 1.15,
    'above_break_3': 1.05,
    'free_throw': 0.85,
}

DEFAULT_WNBA_EFFICIENCY = {
    'rim': 1.15,
    'paint_non_ra': 0.85,
    'midrange': 0.92,
    'corner_3': 1.05,
    'above_break_3': 0.95,
    'free_throw': 0.88,
}


# =============================================================================
# PLAY DEFINITIONS
# =============================================================================

def create_plays():
    """Create simplified play definitions"""
    
    plays = []
    
    # FAV Zoom
    plays.append(Play(
        name="FAV Zoom",
        category="FAVORITE",
        initial_positions={1: (40, 25), 2: (35, 47), 3: (25, 40), 4: (12, 40), 5: (19, 25)},
        branches=[
            PlayBranch("Delay PnR Roll", "rim", 0.30, "Ball screen with roll to basket",
                      {1: (25, 25), 2: (35, 47), 3: (25, 40), 4: (12, 40), 5: (10, 20)}),
            PlayBranch("Delay PnR Pop", "midrange", 0.25, "Screener pops for midrange",
                      {1: (25, 25), 2: (35, 47), 3: (25, 40), 4: (12, 40), 5: (35, 30)}),
            PlayBranch("High Split", "midrange", 0.20, "Cutter gets ball in gap",
                      {1: (40, 25), 2: (35, 47), 3: (15, 35), 4: (12, 40), 5: (19, 25)}),
            PlayBranch("Kick to Corner", "corner_3", 0.25, "Drive and kick to corner",
                      {1: (20, 25), 2: (35, 47), 3: (25, 40), 4: (12, 40), 5: (19, 25)}),
        ],
        description="Delay action into ball screen with multiple reads"
    ))
    
    # FAV Elbow
    plays.append(Play(
        name="FAV Elbow",
        category="FAVORITE",
        initial_positions={1: (30, 47), 2: (40, 35), 3: (40, 15), 4: (10, 35), 5: (19, 25)},
        branches=[
            PlayBranch("Scissors Cut 1", "rim", 0.25, "First cutter to basket",
                      {1: (5, 30), 2: (40, 35), 3: (40, 15), 4: (10, 35), 5: (19, 25)}),
            PlayBranch("Scissors Cut 2", "rim", 0.25, "Second cutter scores",
                      {1: (30, 47), 2: (5, 20), 3: (40, 15), 4: (10, 35), 5: (19, 25)}),
            PlayBranch("5 Face Up", "rim", 0.20, "Big attacks from elbow",
                      {1: (30, 47), 2: (40, 35), 3: (40, 15), 4: (10, 35), 5: (8, 25)}),
            PlayBranch("Pop to Corner 3", "corner_3", 0.30, "Corner shooter spots up",
                      {1: (30, 47), 2: (40, 35), 3: (35, 3), 4: (10, 35), 5: (19, 25)}),
        ],
        description="Classic scissors cut action off elbow"
    ))
    
    # SOB Basic Zoom
    plays.append(Play(
        name="SOB Basic Zoom",
        category="SOB",
        initial_positions={1: (25, 8), 2: (35, 5), 3: (25, 25), 4: (35, 40), 5: (19, 25)},
        branches=[
            PlayBranch("Zipper to Shooter", "above_break_3", 0.35, "3 uses screen for wing 3",
                      {1: (25, 8), 2: (35, 5), 3: (35, 35), 4: (35, 40), 5: (25, 30)}),
            PlayBranch("Screen Pop", "free_throw", 0.25, "Screener pops to elbow",
                      {1: (25, 8), 2: (35, 5), 3: (25, 25), 4: (35, 40), 5: (19, 30)}),
            PlayBranch("Backdoor Cut", "rim", 0.15, "Wing backdoor cuts",
                      {1: (25, 8), 2: (8, 20), 3: (25, 25), 4: (35, 40), 5: (19, 25)}),
            PlayBranch("Reset/Swing", "midrange", 0.25, "Reset offense",
                      {1: (35, 25), 2: (35, 5), 3: (35, 35), 4: (40, 25), 5: (19, 25)}),
        ],
        description="Zipper action to free shooter off screen"
    ))
    
    # SOB Triangle
    plays.append(Play(
        name="SOB Triangle",
        category="SOB",
        initial_positions={1: (25, 8), 2: (40, 3), 3: (25, 25), 4: (12, 35), 5: (19, 25)},
        branches=[
            PlayBranch("Triangle Entry", "rim", 0.30, "Curl to basket off screen",
                      {1: (25, 8), 2: (40, 3), 3: (25, 25), 4: (8, 25), 5: (15, 30)}),
            PlayBranch("Skip to Corner", "corner_3", 0.25, "Skip pass to corner 3",
                      {1: (25, 8), 2: (40, 3), 3: (25, 25), 4: (12, 35), 5: (19, 25)}),
            PlayBranch("Pick the Picker", "midrange", 0.20, "Screener pops to wing",
                      {1: (25, 8), 2: (40, 3), 3: (17, 28), 4: (12, 35), 5: (25, 35)}),
            PlayBranch("Inbounder Iso", "midrange", 0.25, "Inbounder attacks",
                      {1: (35, 25), 2: (40, 3), 3: (25, 25), 4: (12, 35), 5: (19, 25)}),
        ],
        description="Multiple options off triangle formation"
    ))
    
    # BOB Triangle Down
    plays.append(Play(
        name="BOB Triangle Down",
        category="BOB",
        initial_positions={1: (12, 0), 2: (25, 15), 3: (19, 30), 4: (30, 40), 5: (8, 20)},
        branches=[
            PlayBranch("Hit Big on Wing", "midrange", 0.35, "Get ball to big at wing",
                      {1: (12, 0), 2: (25, 15), 3: (12, 25), 4: (30, 40), 5: (25, 25)}),
            PlayBranch("Screen Away Corner", "corner_3", 0.30, "Inbounder screens for corner",
                      {1: (25, 8), 2: (12, 3), 3: (19, 30), 4: (30, 40), 5: (25, 25)}),
            PlayBranch("Lob Option", "rim", 0.15, "Lob to weak side cutter",
                      {1: (12, 0), 2: (25, 15), 3: (19, 30), 4: (8, 28), 5: (8, 20)}),
            PlayBranch("Safety Valve", "midrange", 0.20, "Get ball in and attack",
                      {1: (12, 0), 2: (25, 15), 3: (19, 30), 4: (35, 35), 5: (8, 20)}),
        ],
        description="Late clock baseline out"
    ))
    
    # FAV Nail
    plays.append(Play(
        name="FAV Nail",
        category="FAVORITE",
        initial_positions={1: (40, 25), 2: (35, 3), 3: (12, 15), 4: (12, 40), 5: (25, 25)},
        branches=[
            PlayBranch("Pin Down to PnR", "midrange", 0.30, "Pin down into ball screen",
                      {1: (40, 25), 2: (35, 3), 3: (15, 25), 4: (12, 40), 5: (30, 22)}),
            PlayBranch("Shooter Curl", "rim", 0.20, "Shooter curls tight to basket",
                      {1: (40, 25), 2: (35, 3), 3: (5, 20), 4: (12, 40), 5: (15, 10)}),
            PlayBranch("Fade to Corner", "corner_3", 0.25, "Fade to corner if denied",
                      {1: (40, 25), 2: (35, 3), 3: (8, 3), 4: (12, 40), 5: (25, 25)}),
            PlayBranch("Weak Side Swing", "above_break_3", 0.25, "Ball reversal for 3",
                      {1: (40, 25), 2: (35, 3), 3: (12, 15), 4: (25, 47), 5: (25, 25)}),
        ],
        description="Pin down action to ball screen"
    ))
    
    return plays


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def draw_half_court(ax, court_color='#f5f5f0', line_color='black'):
    """Draw a half basketball court"""
    ax.set_facecolor(court_color)
    
    # Court outline
    ax.add_patch(Rectangle((0, 0), 47, 50, linewidth=2, 
                           edgecolor=line_color, facecolor=court_color))
    
    # Paint
    ax.add_patch(Rectangle((0, 17), 19, 16, linewidth=2,
                           edgecolor=line_color, facecolor='none'))
    
    # Free throw circle
    ax.add_patch(Arc((19, 25), 12, 12, angle=0, theta1=270, theta2=90,
                    linewidth=2, edgecolor=line_color))
    
    # 3-point line
    ax.add_patch(Arc((5.25, 25), 47.5, 47.5, angle=0, theta1=292, theta2=68,
                    linewidth=2, edgecolor=line_color))
    ax.plot([0, 14], [3, 3], color=line_color, linewidth=2)
    ax.plot([0, 14], [47, 47], color=line_color, linewidth=2)
    
    # Basket
    ax.add_patch(Circle((5.25, 25), 0.75, linewidth=2,
                        edgecolor='orange', facecolor='none'))
    
    # Backboard
    ax.plot([4, 4], [22, 28], color=line_color, linewidth=3)
    
    ax.set_xlim(-2, 49)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')
    ax.axis('off')


def visualize_play_branch(play: Play, branch_idx: int, ax):
    """Visualize a specific branch of a play"""
    draw_half_court(ax)
    
    colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71', 4: '#f1c40f', 5: '#9b59b6'}
    
    branch = play.branches[branch_idx]
    
    # Draw initial positions (lighter)
    for player, (x, y) in play.initial_positions.items():
        ax.scatter(x, y, s=300, c=colors[player], alpha=0.3, edgecolors='gray', linewidth=1, zorder=4)
        ax.text(x, y, str(player), ha='center', va='center', fontsize=9, 
                fontweight='bold', color='gray', zorder=5)
    
    # Draw final positions and movement arrows
    for player, (x1, y1) in play.initial_positions.items():
        x2, y2 = branch.player_positions[player]
        
        # Draw movement arrow if position changed
        if (x1, y1) != (x2, y2):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=colors[player], lw=2))
        
        # Draw final position
        ax.scatter(x2, y2, s=400, c=colors[player], edgecolors='black', linewidth=2, zorder=6)
        ax.text(x2, y2, str(player), ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white', zorder=7)
    
    ax.set_title(f"{play.name}: {branch.name}\n{branch.description}", fontsize=11)


def evaluate_play(play: Play, efficiency: Dict) -> pd.DataFrame:
    """Evaluate all branches of a play"""
    results = []
    for branch in play.branches:
        ev = branch.success_prob * efficiency.get(branch.shot_zone, 0.85)
        results.append({
            'Branch': branch.name,
            'Shot Zone': branch.shot_zone,
            'Success Prob': branch.success_prob,
            'Zone Eff': efficiency.get(branch.shot_zone, 0.85),
            'Expected Value': ev
        })
    return pd.DataFrame(results)


# =============================================================================
# STREAMLIT APP
# =============================================================================

st.title("♟️🏀 Chess-Style Play Evaluator")
st.markdown("### Analyzing Phoenix Suns Plays for NBA vs WNBA")

st.markdown("""
Just like chess engines evaluate openings differently based on playing style, 
basketball plays should be evaluated differently based on league-specific efficiency data.
""")

# Initialize plays
plays = create_plays()
play_names = [p.name for p in plays]

# Sidebar
st.sidebar.header("🎮 Controls")

selected_play_name = st.sidebar.selectbox("Select Play", play_names)
selected_play = next(p for p in plays if p.name == selected_play_name)

st.sidebar.markdown("---")
st.sidebar.markdown("### Zone Efficiency Adjustments")

# Allow efficiency adjustments
adjust_efficiency = st.sidebar.checkbox("Custom zone efficiencies", value=False)

if adjust_efficiency:
    st.sidebar.markdown("**NBA Adjustments**")
    nba_rim = st.sidebar.slider("NBA Rim", 0.8, 1.5, 1.30, 0.05)
    nba_mid = st.sidebar.slider("NBA Midrange", 0.5, 1.2, 0.80, 0.05)
    nba_corner = st.sidebar.slider("NBA Corner 3", 0.8, 1.4, 1.15, 0.05)
    nba_above = st.sidebar.slider("NBA Above-break 3", 0.8, 1.3, 1.05, 0.05)
    
    st.sidebar.markdown("**WNBA Adjustments**")
    wnba_rim = st.sidebar.slider("WNBA Rim", 0.8, 1.5, 1.15, 0.05)
    wnba_mid = st.sidebar.slider("WNBA Midrange", 0.5, 1.2, 0.92, 0.05)
    wnba_corner = st.sidebar.slider("WNBA Corner 3", 0.8, 1.4, 1.05, 0.05)
    wnba_above = st.sidebar.slider("WNBA Above-break 3", 0.8, 1.3, 0.95, 0.05)
    
    nba_eff = {'rim': nba_rim, 'midrange': nba_mid, 'corner_3': nba_corner, 
               'above_break_3': nba_above, 'paint_non_ra': 0.82, 'free_throw': 0.85}
    wnba_eff = {'rim': wnba_rim, 'midrange': wnba_mid, 'corner_3': wnba_corner,
                'above_break_3': wnba_above, 'paint_non_ra': 0.85, 'free_throw': 0.88}
else:
    nba_eff = DEFAULT_NBA_EFFICIENCY
    wnba_eff = DEFAULT_WNBA_EFFICIENCY

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Play info
st.subheader(f"📋 {selected_play.name}")
st.markdown(f"**Category:** {selected_play.category} | **Description:** {selected_play.description}")

# Visualize all branches
st.markdown("### Play Diagrams: All Branches")

n_branches = len(selected_play.branches)
cols = st.columns(min(n_branches, 4))

for i, col in enumerate(cols):
    if i < n_branches:
        with col:
            fig, ax = plt.subplots(figsize=(6, 6))
            visualize_play_branch(selected_play, i, ax)
            st.pyplot(fig)
            plt.close()

# Evaluation comparison
st.markdown("---")
st.markdown("### 📊 NBA vs WNBA Evaluation")

eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.markdown("#### NBA Evaluation")
    nba_df = evaluate_play(selected_play, nba_eff)
    nba_total = nba_df['Expected Value'].sum()
    
    st.dataframe(nba_df.style.format({
        'Success Prob': '{:.0%}',
        'Zone Eff': '{:.2f}',
        'Expected Value': '{:.3f}'
    }).background_gradient(subset=['Expected Value'], cmap='Greens'), 
    use_container_width=True)
    
    st.metric("Total Expected Value", f"{nba_total:.3f}")

with eval_col2:
    st.markdown("#### WNBA Evaluation")
    wnba_df = evaluate_play(selected_play, wnba_eff)
    wnba_total = wnba_df['Expected Value'].sum()
    
    st.dataframe(wnba_df.style.format({
        'Success Prob': '{:.0%}',
        'Zone Eff': '{:.2f}',
        'Expected Value': '{:.3f}'
    }).background_gradient(subset=['Expected Value'], cmap='Purples'),
    use_container_width=True)
    
    delta = wnba_total - nba_total
    st.metric("Total Expected Value", f"{wnba_total:.3f}", 
              delta=f"{delta:+.3f} vs NBA",
              delta_color="normal" if delta > 0 else "inverse")

# Decision tree visualization
st.markdown("---")
st.markdown("### 🌳 Decision Tree Comparison")

fig, ax = plt.subplots(figsize=(14, 6))

branches = [b.name[:12] + "..." if len(b.name) > 12 else b.name for b in selected_play.branches]
nba_evs = nba_df['Expected Value'].tolist()
wnba_evs = wnba_df['Expected Value'].tolist()

x = np.arange(len(branches))
width = 0.35

bars1 = ax.bar(x - width/2, nba_evs, width, label='NBA', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + width/2, wnba_evs, width, label='WNBA', color='#9b59b6', alpha=0.8)

ax.set_ylabel('Expected Points')
ax.set_title(f'{selected_play.name}: Branch Expected Values')
ax.set_xticks(x)
ax.set_xticklabels(branches, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars1, nba_evs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#27ae60')
for bar, val in zip(bars2, wnba_evs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#8e44ad')

st.pyplot(fig)
plt.close()

# =============================================================================
# ALL PLAYS COMPARISON
# =============================================================================

st.markdown("---")
st.markdown("### 📈 All Plays Comparison")

all_plays_data = []
for play in plays:
    nba_eval = evaluate_play(play, nba_eff)
    wnba_eval = evaluate_play(play, wnba_eff)
    
    nba_total = nba_eval['Expected Value'].sum()
    wnba_total = wnba_eval['Expected Value'].sum()
    
    all_plays_data.append({
        'Play': play.name,
        'Category': play.category,
        'NBA EV': nba_total,
        'WNBA EV': wnba_total,
        'WNBA Advantage': wnba_total - nba_total
    })

all_plays_df = pd.DataFrame(all_plays_data)

# Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Bar comparison
play_names_short = [p.replace('FAV ', '').replace('SOB ', '').replace('BOB ', '') for p in all_plays_df['Play']]
x = np.arange(len(play_names_short))
width = 0.35

ax1.bar(x - width/2, all_plays_df['NBA EV'], width, label='NBA', color='#2ecc71', alpha=0.8)
ax1.bar(x + width/2, all_plays_df['WNBA EV'], width, label='WNBA', color='#9b59b6', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(play_names_short, rotation=45, ha='right')
ax1.set_ylabel('Expected Points Per Possession')
ax1.set_title('Play Effectiveness by League')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Right: WNBA advantage
colors = ['#9b59b6' if v > 0 else '#e74c3c' for v in all_plays_df['WNBA Advantage']]
ax2.barh(play_names_short, all_plays_df['WNBA Advantage'], color=colors, alpha=0.8)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_xlabel('WNBA Advantage (EPP)')
ax2.set_title('WNBA vs NBA Advantage')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Table
st.dataframe(
    all_plays_df.style.format({
        'NBA EV': '{:.3f}',
        'WNBA EV': '{:.3f}',
        'WNBA Advantage': '{:+.3f}'
    }).background_gradient(subset=['WNBA Advantage'], cmap='RdYlGn', vmin=-0.15, vmax=0.05),
    use_container_width=True
)

# =============================================================================
# INSIGHTS
# =============================================================================

st.markdown("---")
st.markdown("### 🎯 Key Insights")

best_wnba = all_plays_df.loc[all_plays_df['WNBA Advantage'].idxmax()]
worst_wnba = all_plays_df.loc[all_plays_df['WNBA Advantage'].idxmin()]

col1, col2 = st.columns(2)

with col1:
    st.success(f"""
    #### ✅ Best WNBA Play: {best_wnba['Play']}
    
    **WNBA Advantage:** {best_wnba['WNBA Advantage']:+.3f} EPP
    
    This play works better in WNBA because it emphasizes 
    midrange looks, which are +15% more efficient in WNBA.
    """)

with col2:
    st.error(f"""
    #### ⚠️ Worst for WNBA: {worst_wnba['Play']}
    
    **WNBA Disadvantage:** {worst_wnba['WNBA Advantage']:+.3f} EPP
    
    This play relies heavily on rim finishing and/or 
    above-break 3s, which are less efficient in WNBA.
    """)

st.markdown("""
### The Chess Analogy

> *"The best play isn't the one that works in the NBA—it's the one that works for your league's evaluation function."*

Just like the Sicilian Defense is sharp and double-edged while the French Defense is solid and positional, 
basketball plays have different "personalities." WNBA teams should embrace plays that NBA analytics 
consider suboptimal—because the evaluation function is fundamentally different.

**Strategic Recommendations for WNBA:**
1. 🎯 **Boost midrange options** - They're not a "dead zone" in WNBA
2. 🔄 **Add floaters to rim attacks** - Lower finishing rate means fewer forced layups  
3. ⬇️ **De-emphasize above-break 3s** - Convert some to midrange pull-ups
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with matplotlib, pandas, and streamlit | 
    Inspired by chess opening theory
</div>
""", unsafe_allow_html=True)
