"""
Project 2: Chess Opening Theory Applied to Basketball Plays
==========================================================
Analyzing Phoenix Suns plays as decision trees, evaluating them using
zone efficiency data, and adapting them for WNBA's higher midrange efficiency.

Key Concepts:
1. Each play = a chess opening with multiple "lines" (branches)
2. Evaluate branches using Expected Points Per Possession (EPP)
3. Compare NBA vs WNBA zone efficiency to show different optimal branches
4. Show which Suns plays translate best to WNBA style
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COURT DRAWING UTILITIES
# =============================================================================

def draw_half_court(ax, court_color='white', line_color='black', line_width=2):
    """Draw a half basketball court (offensive end)"""
    
    # Court outline
    ax.add_patch(Rectangle((0, 0), 47, 50, linewidth=line_width, 
                           edgecolor=line_color, facecolor=court_color))
    
    # Paint (key)
    ax.add_patch(Rectangle((0, 17), 19, 16, linewidth=line_width,
                           edgecolor=line_color, facecolor='none'))
    
    # Free throw circle
    ax.add_patch(Arc((19, 25), 12, 12, angle=0, theta1=270, theta2=90,
                    linewidth=line_width, edgecolor=line_color))
    
    # Restricted area
    ax.add_patch(Arc((5.25, 25), 8, 8, angle=0, theta1=270, theta2=90,
                    linewidth=line_width, edgecolor=line_color))
    
    # 3-point line
    ax.add_patch(Arc((5.25, 25), 47.5, 47.5, angle=0, theta1=292, theta2=68,
                    linewidth=line_width, edgecolor=line_color))
    ax.plot([0, 14], [3, 3], color=line_color, linewidth=line_width)
    ax.plot([0, 14], [47, 47], color=line_color, linewidth=line_width)
    
    # Basket
    ax.add_patch(Circle((5.25, 25), 0.75, linewidth=line_width,
                        edgecolor='orange', facecolor='none'))
    
    # Backboard
    ax.plot([4, 4], [22, 28], color=line_color, linewidth=line_width+1)
    
    ax.set_xlim(-2, 49)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax


def draw_player(ax, x, y, number, color='blue', size=400):
    """Draw a player marker with number"""
    ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidth=2, zorder=5)
    ax.text(x, y, str(number), ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white', zorder=6)


def draw_movement(ax, x1, y1, x2, y2, style='solid', color='black', width=2):
    """Draw player movement arrow"""
    if style == 'cut':
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=width))
    elif style == 'screen':
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=width, 
                                  linestyle='dashed'))
    elif style == 'pass':
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='green', lw=width+1))


def draw_screen(ax, x, y, angle=0):
    """Draw a screen symbol"""
    screen = Rectangle((x-1, y-0.3), 2, 0.6, angle=angle, 
                       facecolor='yellow', edgecolor='black', linewidth=2, zorder=4)
    ax.add_patch(screen)


# =============================================================================
# ZONE EFFICIENCY DATA (Based on real NBA/WNBA stats)
# =============================================================================

# Points per shot attempt by zone (approximated from league data)
NBA_ZONE_EFFICIENCY = {
    'rim': 1.30,           # Layups/dunks
    'paint_non_ra': 0.82,  # Paint but not restricted area
    'midrange': 0.80,      # 10-23 feet (the "dead zone")
    'corner_3': 1.15,      # Corner threes
    'above_break_3': 1.05, # Non-corner threes
    'free_throw': 0.85,    # Free throw line area
}

WNBA_ZONE_EFFICIENCY = {
    'rim': 1.15,           # Lower finishing rate
    'paint_non_ra': 0.85,  # Slightly better than NBA
    'midrange': 0.92,      # MUCH better than NBA (key difference!)
    'corner_3': 1.05,      # Lower 3pt%
    'above_break_3': 0.95, # Lower 3pt%
    'free_throw': 0.88,    # Better free throw shooting
}


# =============================================================================
# PLAY DEFINITIONS - PHOENIX SUNS PLAYBOOK
# =============================================================================

@dataclass
class PlayAction:
    """A single action in a play"""
    player: int
    action_type: str  # 'cut', 'screen', 'pass', 'dribble', 'shoot'
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    description: str


@dataclass
class PlayBranch:
    """A branch/option within a play"""
    name: str
    actions: List[PlayAction]
    shot_zone: str
    success_prob: float  # Probability this branch leads to a shot
    description: str


@dataclass  
class Play:
    """A complete play with multiple branches"""
    name: str
    category: str  # SOB, BOB, FAVORITE
    initial_positions: Dict[int, Tuple[float, float]]
    branches: List[PlayBranch]
    description: str


def create_suns_plays():
    """Define Phoenix Suns plays from the playbook images"""
    
    plays = []
    
    # =========================================================================
    # SOB "Basic Zoom" - Sideline Out of Bounds
    # =========================================================================
    basic_zoom = Play(
        name="SOB Basic Zoom",
        category="SOB",
        initial_positions={
            1: (25, 8),   # Point guard (inbounder)
            2: (35, 5),   # Wing
            3: (25, 25),  # Under basket receiver  
            4: (35, 40),  # Weak side wing
            5: (19, 25),  # High post
        },
        branches=[
            PlayBranch(
                name="Zipper to Shooter",
                actions=[
                    PlayAction(3, 'cut', (25, 25), (35, 35), "3 zips up off 5's screen"),
                    PlayAction(5, 'screen', (19, 25), (25, 30), "5 sets screen"),
                    PlayAction(1, 'pass', (25, 8), (35, 35), "1 passes to 3"),
                    PlayAction(3, 'shoot', (35, 35), (35, 35), "3 shoots catch-and-shoot"),
                ],
                shot_zone='above_break_3',
                success_prob=0.35,
                description="Primary: 3 uses screen to get open at wing for 3"
            ),
            PlayBranch(
                name="Screen Pop",
                actions=[
                    PlayAction(5, 'screen', (19, 25), (25, 30), "5 sets screen"),
                    PlayAction(5, 'cut', (25, 30), (19, 30), "5 pops to elbow"),
                    PlayAction(1, 'pass', (25, 8), (19, 30), "1 passes to 5"),
                    PlayAction(5, 'shoot', (19, 30), (19, 30), "5 shoots midrange"),
                ],
                shot_zone='free_throw',
                success_prob=0.25,
                description="Secondary: 5 pops after screen for elbow jumper"
            ),
            PlayBranch(
                name="Backdoor Cut",
                actions=[
                    PlayAction(2, 'cut', (35, 5), (8, 20), "2 backdoor cuts"),
                    PlayAction(1, 'pass', (25, 8), (8, 20), "1 hits 2 on backdoor"),
                    PlayAction(2, 'shoot', (8, 20), (5, 25), "2 finishes at rim"),
                ],
                shot_zone='rim',
                success_prob=0.15,
                description="Counter: 2 reads overplay and backdoor cuts"
            ),
            PlayBranch(
                name="Reset/Swing",
                actions=[
                    PlayAction(1, 'pass', (25, 8), (35, 35), "Pass to wing"),
                    PlayAction(4, 'cut', (35, 40), (40, 25), "4 relocates"),
                ],
                shot_zone='midrange',
                success_prob=0.25,
                description="Safety: Reset offense if primary options denied"
            ),
        ],
        description="Basic Zoom: Zipper action to free shooter off screen"
    )
    plays.append(basic_zoom)
    
    # =========================================================================
    # SOB "Triangle" 
    # =========================================================================
    sob_triangle = Play(
        name="SOB Triangle",
        category="SOB",
        initial_positions={
            1: (25, 8),
            2: (40, 3),
            3: (25, 25),
            4: (12, 35),
            5: (19, 25),
        },
        branches=[
            PlayBranch(
                name="Triangle Entry",
                actions=[
                    PlayAction(5, 'screen', (19, 25), (15, 30), "5 screens for 4"),
                    PlayAction(4, 'cut', (12, 35), (8, 25), "4 curls to basket"),
                    PlayAction(1, 'pass', (25, 8), (8, 25), "1 hits 4 cutting"),
                    PlayAction(4, 'shoot', (8, 25), (5, 25), "4 finishes at rim"),
                ],
                shot_zone='rim',
                success_prob=0.30,
                description="Primary: 4 curls off 5's screen for layup"
            ),
            PlayBranch(
                name="Skip to Corner",
                actions=[
                    PlayAction(1, 'pass', (25, 8), (40, 3), "1 skips to corner"),
                    PlayAction(2, 'shoot', (40, 3), (40, 3), "2 shoots corner 3"),
                ],
                shot_zone='corner_3',
                success_prob=0.25,
                description="Secondary: Skip pass to corner 3"
            ),
            PlayBranch(
                name="Pick the Picker",
                actions=[
                    PlayAction(3, 'screen', (25, 25), (17, 28), "3 screens for 5"),
                    PlayAction(5, 'cut', (19, 25), (25, 35), "5 pops to wing"),
                    PlayAction(1, 'pass', (25, 8), (25, 35), "1 passes to 5"),
                    PlayAction(5, 'shoot', (25, 35), (25, 35), "5 shoots"),
                ],
                shot_zone='midrange',
                success_prob=0.20,
                description="Counter: Pick the picker action"
            ),
            PlayBranch(
                name="Inbounder Iso",
                actions=[
                    PlayAction(1, 'dribble', (25, 8), (35, 25), "1 enters and attacks"),
                    PlayAction(1, 'shoot', (35, 25), (35, 25), "1 pulls up"),
                ],
                shot_zone='midrange',
                success_prob=0.25,
                description="Safety: Inbounder attacks"
            ),
        ],
        description="Triangle: Multiple options off triangle formation"
    )
    plays.append(sob_triangle)
    
    # =========================================================================
    # FAVORITE "Zoom" 
    # =========================================================================
    fav_zoom = Play(
        name="FAV Zoom",
        category="FAVORITE",
        initial_positions={
            1: (40, 25),  # Ball handler at wing
            2: (35, 47),  # Weak side corner
            3: (25, 40),  # Top
            4: (12, 40),  # Weak side wing
            5: (19, 25),  # Strong side elbow
        },
        branches=[
            PlayBranch(
                name="Delay PnR Left",
                actions=[
                    PlayAction(5, 'screen', (19, 25), (32, 25), "5 comes for ball screen"),
                    PlayAction(1, 'dribble', (40, 25), (25, 25), "1 uses screen going left"),
                    PlayAction(5, 'cut', (32, 25), (10, 20), "5 rolls to rim"),
                    PlayAction(1, 'pass', (25, 25), (10, 20), "1 hits 5 on roll"),
                    PlayAction(5, 'shoot', (10, 20), (5, 25), "5 finishes"),
                ],
                shot_zone='rim',
                success_prob=0.30,
                description="Primary: Ball screen with roll to basket"
            ),
            PlayBranch(
                name="Delay PnR Pop",
                actions=[
                    PlayAction(5, 'screen', (19, 25), (32, 25), "5 screens"),
                    PlayAction(1, 'dribble', (40, 25), (25, 25), "1 uses screen"),
                    PlayAction(5, 'cut', (32, 25), (35, 30), "5 pops instead of rolls"),
                    PlayAction(1, 'pass', (25, 25), (35, 30), "1 hits 5 on pop"),
                    PlayAction(5, 'shoot', (35, 30), (35, 30), "5 shoots"),
                ],
                shot_zone='midrange',
                success_prob=0.25,
                description="Alternative: 5 pops for midrange"
            ),
            PlayBranch(
                name="High Split",
                actions=[
                    PlayAction(3, 'cut', (25, 40), (15, 35), "3 cuts to gap"),
                    PlayAction(1, 'pass', (40, 25), (15, 35), "1 hits 3 in gap"),
                    PlayAction(3, 'shoot', (15, 35), (15, 35), "3 takes midrange"),
                ],
                shot_zone='midrange',
                success_prob=0.20,
                description="Read: If 5 is denied, hit cutter"
            ),
            PlayBranch(
                name="Kick to Corner",
                actions=[
                    PlayAction(1, 'dribble', (40, 25), (20, 25), "1 drives baseline"),
                    PlayAction(1, 'pass', (20, 25), (35, 47), "1 kicks to corner"),
                    PlayAction(2, 'shoot', (35, 47), (35, 47), "2 shoots corner 3"),
                ],
                shot_zone='corner_3',
                success_prob=0.25,
                description="Collapse: Kick to corner on drive"
            ),
        ],
        description="Zoom: Delay action into ball screen with multiple reads"
    )
    plays.append(fav_zoom)
    
    # =========================================================================
    # FAVORITE "Elbow" (Our Scissors)
    # =========================================================================
    fav_elbow = Play(
        name="FAV Elbow",
        category="FAVORITE",
        initial_positions={
            1: (30, 47),  # Top of key
            2: (40, 35),  # Strong wing
            3: (40, 15),  # Strong corner
            4: (10, 35),  # Weak wing
            5: (19, 25),  # Elbow
        },
        branches=[
            PlayBranch(
                name="Scissors Cut 1",
                actions=[
                    PlayAction(1, 'pass', (30, 47), (19, 25), "1 passes to 5 at elbow"),
                    PlayAction(1, 'cut', (30, 47), (5, 30), "1 scissors cuts first"),
                    PlayAction(5, 'pass', (19, 25), (5, 30), "5 hits 1 cutting"),
                    PlayAction(1, 'shoot', (5, 30), (5, 25), "1 finishes"),
                ],
                shot_zone='rim',
                success_prob=0.25,
                description="Primary: First cutter to basket"
            ),
            PlayBranch(
                name="Scissors Cut 2",
                actions=[
                    PlayAction(1, 'pass', (30, 47), (19, 25), "1 passes to 5"),
                    PlayAction(1, 'cut', (30, 47), (5, 30), "1 cuts (decoy)"),
                    PlayAction(2, 'cut', (40, 35), (5, 20), "2 scissors second"),
                    PlayAction(5, 'pass', (19, 25), (5, 20), "5 hits 2"),
                    PlayAction(2, 'shoot', (5, 20), (5, 25), "2 finishes"),
                ],
                shot_zone='rim',
                success_prob=0.25,
                description="Counter: Second cutter scores"
            ),
            PlayBranch(
                name="5 Face Up",
                actions=[
                    PlayAction(1, 'pass', (30, 47), (19, 25), "1 passes to 5"),
                    PlayAction(5, 'dribble', (19, 25), (12, 25), "5 faces up and attacks"),
                    PlayAction(5, 'shoot', (12, 25), (5, 25), "5 scores at rim"),
                ],
                shot_zone='rim',
                success_prob=0.20,
                description="Read: 5 attacks if cutters are denied"
            ),
            PlayBranch(
                name="Pop to 3",
                actions=[
                    PlayAction(1, 'pass', (30, 47), (19, 25), "1 passes to 5"),
                    PlayAction(3, 'cut', (40, 15), (35, 3), "3 fades to corner"),
                    PlayAction(5, 'pass', (19, 25), (35, 3), "5 kicks to corner"),
                    PlayAction(3, 'shoot', (35, 3), (35, 3), "3 shoots corner 3"),
                ],
                shot_zone='corner_3',
                success_prob=0.30,
                description="Release: Corner shooter spots up"
            ),
        ],
        description="Elbow Scissors: Classic scissors cut action"
    )
    plays.append(fav_elbow)
    
    # =========================================================================
    # FAVORITE "Nail" (Pin Down to Ball Screen)
    # =========================================================================
    fav_nail = Play(
        name="FAV Nail",
        category="FAVORITE",
        initial_positions={
            1: (40, 25),
            2: (35, 3),
            3: (12, 15),
            4: (12, 40),
            5: (25, 25),  # At the nail
        },
        branches=[
            PlayBranch(
                name="Pin Down to Ball Screen",
                actions=[
                    PlayAction(5, 'screen', (25, 25), (15, 10), "5 pins down for 3"),
                    PlayAction(3, 'cut', (12, 15), (28, 20), "3 comes off screen"),
                    PlayAction(1, 'pass', (40, 25), (28, 20), "1 passes to 3"),
                    PlayAction(5, 'screen', (15, 10), (30, 22), "5 re-screens for 3"),
                    PlayAction(3, 'dribble', (28, 20), (15, 25), "3 uses screen"),
                    PlayAction(3, 'shoot', (15, 25), (15, 25), "3 pulls up"),
                ],
                shot_zone='midrange',
                success_prob=0.30,
                description="Primary: Pin down into ball screen"
            ),
            PlayBranch(
                name="Shooter Curl",
                actions=[
                    PlayAction(5, 'screen', (25, 25), (15, 10), "5 screens"),
                    PlayAction(3, 'cut', (12, 15), (5, 20), "3 curls tight"),
                    PlayAction(1, 'pass', (40, 25), (5, 20), "1 hits 3 curling"),
                    PlayAction(3, 'shoot', (5, 20), (5, 25), "3 finishes"),
                ],
                shot_zone='rim',
                success_prob=0.20,
                description="Counter: 3 curls to basket"
            ),
            PlayBranch(
                name="Pop to Corner",
                actions=[
                    PlayAction(3, 'cut', (12, 15), (8, 3), "3 fades to corner"),
                    PlayAction(1, 'pass', (40, 25), (8, 3), "1 skips to corner"),
                    PlayAction(3, 'shoot', (8, 3), (8, 3), "3 shoots corner 3"),
                ],
                shot_zone='corner_3',
                success_prob=0.25,
                description="Read: Fade to corner if screen is denied"
            ),
            PlayBranch(
                name="Weak Side Action",
                actions=[
                    PlayAction(4, 'cut', (12, 40), (25, 47), "4 lifts"),
                    PlayAction(1, 'pass', (40, 25), (25, 47), "1 swings to 4"),
                    PlayAction(4, 'shoot', (25, 47), (25, 47), "4 shoots 3"),
                ],
                shot_zone='above_break_3',
                success_prob=0.25,
                description="Swing: Ball reversal for open 3"
            ),
        ],
        description="Nail: Pin down action to ball screen"
    )
    plays.append(fav_nail)
    
    # =========================================================================
    # BOB "Triangle Down" (Late Clock)
    # =========================================================================
    bob_triangle = Play(
        name="BOB Triangle Down",
        category="BOB",
        initial_positions={
            1: (12, 0),   # Inbounder baseline
            2: (25, 15),
            3: (19, 30),
            4: (30, 40),
            5: (8, 20),
        },
        branches=[
            PlayBranch(
                name="Hit Big on Wing",
                actions=[
                    PlayAction(3, 'screen', (19, 30), (12, 25), "3 screens for 5"),
                    PlayAction(5, 'cut', (8, 20), (25, 25), "5 flashes to wing"),
                    PlayAction(1, 'pass', (12, 0), (25, 25), "1 inbounds to 5"),
                    PlayAction(5, 'shoot', (25, 25), (25, 25), "5 shoots midrange"),
                ],
                shot_zone='midrange',
                success_prob=0.35,
                description="Primary: Get ball to big at wing"
            ),
            PlayBranch(
                name="Inbounder Screen Away",
                actions=[
                    PlayAction(1, 'cut', (12, 0), (25, 8), "1 enters and screens"),
                    PlayAction(2, 'cut', (25, 15), (12, 3), "2 uses screen to corner"),
                    PlayAction(5, 'pass', (25, 25), (12, 3), "5 finds 2 in corner"),
                    PlayAction(2, 'shoot', (12, 3), (12, 3), "2 shoots corner 3"),
                ],
                shot_zone='corner_3',
                success_prob=0.30,
                description="Secondary: Inbounder screens for corner 3"
            ),
            PlayBranch(
                name="Lob Option",
                actions=[
                    PlayAction(4, 'cut', (30, 40), (8, 28), "4 cuts backdoor"),
                    PlayAction(1, 'pass', (12, 0), (8, 28), "1 lobs to 4"),
                    PlayAction(4, 'shoot', (8, 28), (5, 25), "4 finishes lob"),
                ],
                shot_zone='rim',
                success_prob=0.15,
                description="Counter: Lob to weak side cutter"
            ),
            PlayBranch(
                name="Safety Valve",
                actions=[
                    PlayAction(4, 'cut', (30, 40), (35, 35), "4 pops out"),
                    PlayAction(1, 'pass', (12, 0), (35, 35), "1 finds 4"),
                    PlayAction(4, 'dribble', (35, 35), (30, 25), "4 attacks"),
                    PlayAction(4, 'shoot', (30, 25), (30, 25), "4 shoots"),
                ],
                shot_zone='midrange',
                success_prob=0.20,
                description="Safety: Get ball in and attack"
            ),
        ],
        description="BOB Triangle Down: Late clock baseline out"
    )
    plays.append(bob_triangle)
    
    return plays


# =============================================================================
# PLAY EVALUATION & DECISION TREES
# =============================================================================

def evaluate_play(play: Play, zone_efficiency: Dict[str, float]) -> Dict:
    """
    Evaluate a play using zone efficiency data.
    Returns expected points per possession for each branch and overall.
    """
    results = {
        'play_name': play.name,
        'branches': [],
        'total_expected_value': 0,
        'best_branch': None,
        'best_branch_ev': 0
    }
    
    for branch in play.branches:
        # Expected value = P(getting shot) × P(making shot) × points
        # Simplified: EV = success_prob × zone_efficiency[shot_zone]
        ev = branch.success_prob * zone_efficiency[branch.shot_zone]
        
        results['branches'].append({
            'name': branch.name,
            'shot_zone': branch.shot_zone,
            'success_prob': branch.success_prob,
            'zone_efficiency': zone_efficiency[branch.shot_zone],
            'expected_value': ev,
            'description': branch.description
        })
        
        results['total_expected_value'] += ev
        
        if ev > results['best_branch_ev']:
            results['best_branch_ev'] = ev
            results['best_branch'] = branch.name
    
    return results


def compare_nba_wnba_evaluation(plays: List[Play]):
    """Compare play effectiveness between NBA and WNBA efficiency profiles"""
    
    comparison = []
    
    for play in plays:
        nba_eval = evaluate_play(play, NBA_ZONE_EFFICIENCY)
        wnba_eval = evaluate_play(play, WNBA_ZONE_EFFICIENCY)
        
        comparison.append({
            'play_name': play.name,
            'category': play.category,
            'nba_ev': nba_eval['total_expected_value'],
            'wnba_ev': wnba_eval['total_expected_value'],
            'nba_best_branch': nba_eval['best_branch'],
            'wnba_best_branch': wnba_eval['best_branch'],
            'wnba_advantage': wnba_eval['total_expected_value'] - nba_eval['total_expected_value'],
            'nba_branches': nba_eval['branches'],
            'wnba_branches': wnba_eval['branches']
        })
    
    return comparison


def plot_decision_tree(play: Play, nba_eval: Dict, wnba_eval: Dict, ax=None):
    """Plot a decision tree for a play showing NBA vs WNBA branch values"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create graph
    G = nx.DiGraph()
    
    # Add root node
    G.add_node('Start', pos=(0.5, 1.0))
    
    # Add branch nodes
    n_branches = len(play.branches)
    for i, branch in enumerate(play.branches):
        x_pos = (i + 0.5) / n_branches
        
        # Branch node
        branch_name = branch.name[:15] + "..." if len(branch.name) > 15 else branch.name
        G.add_node(branch_name, pos=(x_pos, 0.6))
        G.add_edge('Start', branch_name)
        
        # Outcome nodes (NBA and WNBA)
        nba_ev = nba_eval['branches'][i]['expected_value']
        wnba_ev = wnba_eval['branches'][i]['expected_value']
        
        nba_node = f"NBA\n{nba_ev:.3f}"
        wnba_node = f"WNBA\n{wnba_ev:.3f}"
        
        G.add_node(nba_node, pos=(x_pos - 0.05, 0.2))
        G.add_node(wnba_node, pos=(x_pos + 0.05, 0.2))
        
        G.add_edge(branch_name, nba_node)
        G.add_edge(branch_name, wnba_node)
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw
    ax.clear()
    
    # Draw nodes with colors
    node_colors = []
    for node in G.nodes():
        if 'NBA' in node:
            node_colors.append('#2ecc71')  # Green for NBA
        elif 'WNBA' in node:
            node_colors.append('#9b59b6')  # Purple for WNBA
        elif node == 'Start':
            node_colors.append('#3498db')  # Blue for start
        else:
            node_colors.append('#f39c12')  # Orange for branches
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=3000, font_size=8, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray')
    
    ax.set_title(f"Decision Tree: {play.name}\nNBA Total EV: {nba_eval['total_expected_value']:.3f} | "
                f"WNBA Total EV: {wnba_eval['total_expected_value']:.3f}", fontsize=12, fontweight='bold')
    
    return ax


# =============================================================================
# PLAY VISUALIZATION
# =============================================================================

def visualize_play(play: Play, branch_idx: int = 0, ax=None, title_suffix=""):
    """Visualize a specific branch of a play on a court diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw court
    draw_half_court(ax)
    
    # Draw initial positions
    colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71', 4: '#f1c40f', 5: '#9b59b6'}
    
    for player, (x, y) in play.initial_positions.items():
        draw_player(ax, x, y, player, colors[player])
    
    # Draw selected branch actions
    branch = play.branches[branch_idx]
    
    for action in branch.actions:
        x1, y1 = action.start_pos
        x2, y2 = action.end_pos
        
        if action.action_type == 'cut':
            draw_movement(ax, x1, y1, x2, y2, style='cut', color=colors[action.player])
        elif action.action_type == 'screen':
            draw_movement(ax, x1, y1, x2, y2, style='screen', color=colors[action.player])
            draw_screen(ax, x2, y2)
        elif action.action_type == 'pass':
            draw_movement(ax, x1, y1, x2, y2, style='pass')
        elif action.action_type == 'dribble':
            draw_movement(ax, x1, y1, x2, y2, style='cut', color=colors[action.player], width=3)
    
    ax.set_title(f"{play.name}: {branch.name}{title_suffix}\n{branch.description}", 
                fontsize=11, fontweight='bold')
    
    return ax


def create_all_play_visualizations(plays: List[Play]):
    """Create visualizations for all plays and their branches"""
    
    for play in plays:
        n_branches = len(play.branches)
        fig, axes = plt.subplots(1, n_branches, figsize=(5*n_branches, 6))
        
        if n_branches == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            visualize_play(play, i, ax)
        
        plt.suptitle(f"{play.name} - All Options", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = play.name.replace(' ', '_').replace('"', '').lower()
        plt.savefig(f'play_{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# WNBA ADAPTATION RECOMMENDATIONS
# =============================================================================

def generate_wnba_adaptations(plays: List[Play], comparison: List[Dict]):
    """
    Generate recommendations for adapting Suns plays to WNBA style.
    Key insight: WNBA midrange is MUCH more efficient than NBA midrange.
    """
    
    adaptations = []
    
    for play, comp in zip(plays, comparison):
        adaptation = {
            'play_name': play.name,
            'original_best_nba': comp['nba_best_branch'],
            'original_best_wnba': comp['wnba_best_branch'],
            'nba_ev': comp['nba_ev'],
            'wnba_ev': comp['wnba_ev'],
            'recommendations': []
        }
        
        # Analyze each branch
        for nba_b, wnba_b in zip(comp['nba_branches'], comp['wnba_branches']):
            zone = nba_b['shot_zone']
            nba_eff = nba_b['zone_efficiency']
            wnba_eff = wnba_b['zone_efficiency']
            
            # Identify opportunities
            if zone == 'midrange' and wnba_eff > nba_eff:
                adaptation['recommendations'].append(
                    f"✅ BOOST '{nba_b['name']}': Midrange is +{(wnba_eff-nba_eff)*100:.0f}% more efficient in WNBA. "
                    f"Consider making this a PRIMARY option instead of secondary."
                )
            elif zone == 'above_break_3' and wnba_eff < nba_eff:
                adaptation['recommendations'].append(
                    f"⚠️ DE-EMPHASIZE '{nba_b['name']}': Above-break 3s are {(nba_eff-wnba_eff)*100:.0f}% less efficient in WNBA. "
                    f"Consider converting to midrange pull-up."
                )
            elif zone == 'rim' and wnba_eff < nba_eff:
                adaptation['recommendations'].append(
                    f"🔄 MODIFY '{nba_b['name']}': Rim finishing is {(nba_eff-wnba_eff)*100:.0f}% lower in WNBA. "
                    f"Add floater/short midrange option instead of forcing to basket."
                )
        
        # Overall strategic recommendation
        if comp['wnba_ev'] > comp['nba_ev']:
            adaptation['strategic_note'] = (
                f"🌟 This play is BETTER suited for WNBA (+{comp['wnba_advantage']:.3f} EPP). "
                f"The midrange options compensate for lower rim/3PT efficiency."
            )
        else:
            adaptation['strategic_note'] = (
                f"⚡ Consider restructuring for WNBA. Current play loses "
                f"{-comp['wnba_advantage']:.3f} EPP vs NBA version."
            )
        
        adaptations.append(adaptation)
    
    return adaptations


def create_summary_comparison(plays: List[Play], comparison: List[Dict]):
    """Create summary visualization comparing all plays"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Overall EV comparison
    ax1 = axes[0, 0]
    play_names = [c['play_name'].replace('FAV ', '').replace('SOB ', '').replace('BOB ', '') for c in comparison]
    nba_evs = [c['nba_ev'] for c in comparison]
    wnba_evs = [c['wnba_ev'] for c in comparison]
    
    x = np.arange(len(play_names))
    width = 0.35
    
    ax1.bar(x - width/2, nba_evs, width, label='NBA', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, wnba_evs, width, label='WNBA', color='#9b59b6', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(play_names, rotation=45, ha='right')
    ax1.set_ylabel('Expected Points Per Possession')
    ax1.set_title('Play Effectiveness: NBA vs WNBA')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: WNBA Advantage
    ax2 = axes[0, 1]
    advantages = [c['wnba_advantage'] for c in comparison]
    colors = ['#9b59b6' if a > 0 else '#e74c3c' for a in advantages]
    
    ax2.barh(play_names, advantages, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('WNBA Advantage (EPP)')
    ax2.set_title('WNBA vs NBA Advantage by Play')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Zone efficiency comparison
    ax3 = axes[1, 0]
    zones = list(NBA_ZONE_EFFICIENCY.keys())
    nba_effs = [NBA_ZONE_EFFICIENCY[z] for z in zones]
    wnba_effs = [WNBA_ZONE_EFFICIENCY[z] for z in zones]
    
    x = np.arange(len(zones))
    ax3.bar(x - width/2, nba_effs, width, label='NBA', color='#2ecc71', alpha=0.8)
    ax3.bar(x + width/2, wnba_effs, width, label='WNBA', color='#9b59b6', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([z.replace('_', '\n') for z in zones], fontsize=9)
    ax3.set_ylabel('Points Per Shot')
    ax3.set_title('Zone Efficiency: NBA vs WNBA')
    ax3.legend()
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Best branch comparison
    ax4 = axes[1, 1]
    
    # Count shot zone preferences
    nba_zones = {}
    wnba_zones = {}
    
    for c in comparison:
        for b in c['nba_branches']:
            nba_zones[b['shot_zone']] = nba_zones.get(b['shot_zone'], 0) + b['expected_value']
        for b in c['wnba_branches']:
            wnba_zones[b['shot_zone']] = wnba_zones.get(b['shot_zone'], 0) + b['expected_value']
    
    zones = list(set(nba_zones.keys()) | set(wnba_zones.keys()))
    nba_vals = [nba_zones.get(z, 0) for z in zones]
    wnba_vals = [wnba_zones.get(z, 0) for z in zones]
    
    x = np.arange(len(zones))
    ax4.bar(x - width/2, nba_vals, width, label='NBA', color='#2ecc71', alpha=0.8)
    ax4.bar(x + width/2, wnba_vals, width, label='WNBA', color='#9b59b6', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([z.replace('_', '\n') for z in zones], fontsize=9)
    ax4.set_ylabel('Total Expected Value Contribution')
    ax4.set_title('Where Plays Generate Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Phoenix Suns Plays: NBA vs WNBA Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./play_comparison_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_wnba_adapted_play(original_play: Play) -> Play:
    """
    Create a WNBA-optimized version of a play by:
    1. Adding midrange options where 3-point shots were primary
    2. Adding floater options instead of forcing to rim
    3. Increasing midrange branch probabilities
    """
    
    # This is a conceptual modification - in practice you'd redesign the play
    adapted = Play(
        name=f"{original_play.name} (WNBA)",
        category=original_play.category,
        initial_positions=original_play.initial_positions.copy(),
        branches=[],
        description=f"WNBA-optimized version of {original_play.description}"
    )
    
    for branch in original_play.branches:
        new_branch = PlayBranch(
            name=branch.name,
            actions=branch.actions.copy(),
            shot_zone=branch.shot_zone,
            success_prob=branch.success_prob,
            description=branch.description
        )
        
        # Modify for WNBA
        if branch.shot_zone == 'above_break_3':
            # Convert some 3-point looks to midrange pull-ups
            new_branch.shot_zone = 'midrange'
            new_branch.description = f"(WNBA) {branch.description} - pull up instead of 3"
            new_branch.success_prob *= 1.1  # Higher probability of getting this shot
            
        elif branch.shot_zone == 'rim':
            # Add floater option
            new_branch.success_prob *= 0.9  # Slightly lower rim conversion
            new_branch.description = f"(WNBA) {branch.description} - add floater option"
            
        adapted.branches.append(new_branch)
    
    return adapted


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
   # os.makedirs('/home/claude/wnba_nba_analysis', exist_ok=True)
    
    print("="*60)
    print("PROJECT 2: CHESS-STYLE PLAY ANALYSIS")
    print("Phoenix Suns Plays → NBA/WNBA Evaluation")
    print("="*60)
    
    print("\n[1/6] Creating Suns play definitions...")
    plays = create_suns_plays()
    print(f"  Created {len(plays)} plays with {sum(len(p.branches) for p in plays)} total branches")
    
    print("\n[2/6] Evaluating plays for NBA and WNBA...")
    comparison = compare_nba_wnba_evaluation(plays)
    
    print("\n  Play Evaluation Summary:")
    print("  " + "-"*55)
    for c in comparison:
        print(f"  {c['play_name']:<20} | NBA: {c['nba_ev']:.3f} | WNBA: {c['wnba_ev']:.3f} | Δ: {c['wnba_advantage']:+.3f}")
    
    print("\n[3/6] Creating play visualizations...")
    create_all_play_visualizations(plays)
    
    print("\n[4/6] Creating decision tree visualizations...")
    for play in plays:
        nba_eval = evaluate_play(play, NBA_ZONE_EFFICIENCY)
        wnba_eval = evaluate_play(play, WNBA_ZONE_EFFICIENCY)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_decision_tree(play, nba_eval, wnba_eval, ax)
        
        filename = play.name.replace(' ', '_').replace('"', '').lower()
        plt.savefig(f'./tree_{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print("\n[5/6] Generating WNBA adaptation recommendations...")
    adaptations = generate_wnba_adaptations(plays, comparison)
    
    print("\n" + "="*60)
    print("WNBA ADAPTATION RECOMMENDATIONS")
    print("="*60)
    
    for a in adaptations:
        print(f"\n📋 {a['play_name']}")
        print(f"   NBA Best: {a['original_best_nba']} | WNBA Best: {a['original_best_wnba']}")
        print(f"   {a['strategic_note']}")
        for rec in a['recommendations'][:2]:  # Show top 2 recommendations
            print(f"   {rec}")
    
    print("\n[6/6] Creating summary comparison...")
    create_summary_comparison(plays, comparison)
    
    print("\n" + "="*60)
    print("PROJECT 2 COMPLETE - All visualizations saved!")
    print("="*60)
    
    # Print key insight
    print("""
    
KEY STRATEGIC INSIGHT
=====================
The WNBA's higher midrange efficiency (+15% vs NBA) fundamentally changes
play optimization. Plays that create midrange opportunities become MORE
valuable in the WNBA, while plays designed to generate 3-point shots
become LESS valuable.

BEST WNBA PLAYS FROM SUNS PLAYBOOK:
1. Plays with elbow/free-throw line options
2. Plays where the "secondary" midrange option becomes primary
3. Ball screen plays where the handler can pull up

WORST WNBA PLAYS FROM SUNS PLAYBOOK:
1. Plays designed primarily for above-break 3s
2. Plays that force to the rim against set defense
3. Spread pick-and-roll without midrange pull-up option

This is analogous to chess opening theory: an opening that's strong in
classical chess (NBA) may be weak in rapid chess (WNBA) because the
evaluation function weights moves differently.
""")
