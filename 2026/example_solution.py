"""
EXAMPLE SOLUTION - Stranger Things XODIA Challenge
===================================================
IMPROVED VERSION - Much better learning for all 3 universes!

Structure:
- Universe 1: Rescue Will (6 Demogorgons, intelligent pathfinding)
- Universe 2: Escape Room (7 Guards, smart navigation + avoidance)  
- Universe 3: Dual rescue (6 Demogorgons + Vecna, strategic rescue order)
"""

import numpy as np
import random


# ====================================================================
# UNIVERSE 1: RESCUE WILL - IMPROVED AGENT
# ====================================================================

def discretize_universe1(observation):
    """
    Smart discretization for Universe 1
    
    Observation: [agent_x, agent_y, will_x, will_y, demo_x, demo_y]
    
    Focus: Direction to Will + threat awareness
    """
    agent_x, agent_y = observation[0], observation[1]
    will_x, will_y = observation[2], observation[3]
    demo_x, demo_y = observation[4], observation[5]
    
    # Coarser agent position (reduce state space)
    discrete_agent_x = int(agent_x / 3.0)  # 0-6 buckets
    discrete_agent_y = int(agent_y / 3.0)
    
    # Direction to Will (key for pathfinding)
    dx = will_x - agent_x
    dy = will_y - agent_y
    
    # Bucket the direction
    dx_bucket = 0 if abs(dx) < 2 else (1 if dx > 0 else -1)
    dy_bucket = 0 if abs(dy) < 2 else (1 if dy > 0 else -1)
    
    # Distance to Will (bucketed)
    dist_to_will = np.linalg.norm([dx, dy])
    dist_bucket = min(int(dist_to_will / 4.0), 4)  # 0-4
    
    # Nearest Demogorgon threat level
    demo_dist = np.linalg.norm([demo_x - agent_x, demo_y - agent_y])
    threat = 0 if demo_dist > 5 else (1 if demo_dist > 3 else 2)  # 0=safe, 1=caution, 2=danger
    
    return (discrete_agent_x, discrete_agent_y, dx_bucket, dy_bucket, dist_bucket, threat)


def create_universe1_agent():
    """Improved agent for Universe 1: Smart pathfinding to Will (6 Demogorgons)"""
    
    Q1 = {}
    
    # Better hyperparameters for Universe 1
    LEARNING_RATE = 0.5  # Higher for faster learning
    DISCOUNT_FACTOR = 0.95
    exploration = {'epsilon': 0.4, 'decay': 0.997, 'min': 0.05}
    
    def select_action(observation):
        state = discretize_universe1(observation)
        
        # Epsilon-greedy with better exploration
        if random.random() < exploration['epsilon']:
            # Smarter exploration - bias toward Will's direction
            agent_x, agent_y = observation[0], observation[1]
            will_x, will_y = observation[2], observation[3]
            
            dx = will_x - agent_x
            dy = will_y - agent_y
            
            # 70% chance to explore in the right general direction
            if random.random() < 0.7:
                if abs(dx) > abs(dy):
                    return 0 if dx > 0 else 2  # Right or Left
                else:
                    return 1 if dy > 0 else 3  # Up or Down
            else:
                return random.randint(0, 3)
        
        # Exploit: choose best action
        q_values = [Q1.get((state, a), 0.0) for a in range(4)]
        max_q = max(q_values)
        best_actions = [a for a in range(4) if q_values[a] == max_q]
        return random.choice(best_actions)
    
    def update(state, action_index, reward, next_state):
        old_q = Q1.get((state, action_index), 0.0)
        future_q_values = [Q1.get((next_state, a), 0.0) for a in range(4)]
        max_future_q = max(future_q_values)
        
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q)
        Q1[(state, action_index)] = new_q
        
        # Decay exploration
        exploration['epsilon'] *= exploration['decay']
        exploration['epsilon'] = max(exploration['min'], exploration['epsilon'])
    
    return select_action, update, Q1


# ====================================================================
# UNIVERSE 2: ESCAPE ROOM - IMPROVED AGENT
# ====================================================================

def discretize_universe2(observation):
    """
    Smart discretization for Universe 2: Lab Escape
    
    Observation: [agent_x, agent_y, exit_x, exit_y, guard_x, guard_y]
    
    Strategy: Path to exit + guard awareness
    """
    agent_x, agent_y = observation[0], observation[1]
    exit_x, exit_y = observation[2], observation[3]
    guard_x, guard_y = observation[4], observation[5]
    
    # Agent position (moderate granularity for maze)
    discrete_agent_x = int(agent_x / 2.5)  # 0-8 buckets
    discrete_agent_y = int(agent_y / 2.5)
    
    # Direction to exit (critical for navigation)
    dx = exit_x - agent_x
    dy = exit_y - agent_y
    
    # Directional buckets
    dx_bucket = 0 if abs(dx) < 2 else (1 if dx > 0 else -1)
    dy_bucket = 0 if abs(dy) < 2 else (1 if dy > 0 else -1)
    
    # Distance to exit (bucketed)
    dist_to_exit = np.linalg.norm([dx, dy])
    exit_dist_bucket = min(int(dist_to_exit / 3.0), 6)  # 0-6
    
    # Guard proximity (for avoidance)
    guard_dist = np.linalg.norm([guard_x - agent_x, guard_y - agent_y])
    guard_threat = 0 if guard_dist > 5 else (1 if guard_dist > 3 else 2)
    
    return (discrete_agent_x, discrete_agent_y, dx_bucket, dy_bucket, 
            exit_dist_bucket, guard_threat)


def create_universe2_agent():
    """Improved agent for Universe 2: Smart maze navigation (7 Guards)"""
    
    Q2 = {}
    
    # Better hyperparameters for maze
    LEARNING_RATE = 0.55  # Even higher for complex navigation
    DISCOUNT_FACTOR = 0.93
    exploration = {'epsilon': 0.5, 'decay': 0.9985, 'min': 0.08}
    
    def select_action(observation):
        state = discretize_universe2(observation)
        
        if random.random() < exploration['epsilon']:
            # Smart exploration - bias toward exit
            agent_x, agent_y = observation[0], observation[1]
            exit_x, exit_y = observation[2], observation[3]
            
            dx = exit_x - agent_x
            dy = exit_y - agent_y
            
            # 75% chance to move toward exit during exploration
            if random.random() < 0.75:
                if abs(dx) > abs(dy):
                    return 0 if dx > 0 else 2
                else:
                    return 1 if dy > 0 else 3
            else:
                return random.randint(0, 3)
        
        q_values = [Q2.get((state, a), 0.0) for a in range(4)]
        max_q = max(q_values)
        best_actions = [a for a in range(4) if q_values[a] == max_q]
        return random.choice(best_actions)
    
    def update(state, action_index, reward, next_state):
        old_q = Q2.get((state, action_index), 0.0)
        future_q_values = [Q2.get((next_state, a), 0.0) for a in range(4)]
        max_future_q = max(future_q_values)
        
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q)
        Q2[(state, action_index)] = new_q
        
        exploration['epsilon'] *= exploration['decay']
        exploration['epsilon'] = max(exploration['min'], exploration['epsilon'])
    
    return select_action, update, Q2


# ====================================================================
# UNIVERSE 3: DEFEAT VECNA - IMPROVED AGENT
# ====================================================================

def discretize_universe3(observation):
    """
    Smart discretization for Universe 3: Dual rescue mission
    
    Observation: [agent_x, agent_y, child1_x, child1_y, child2_x, child2_y,
                  vecna_x, vecna_y, child1_rescued, child2_rescued, proj_x, proj_y]
    
    Strategy: Prioritize nearest child + avoid threats
    """
    agent_x, agent_y = observation[0], observation[1]
    child1_x, child1_y = observation[2], observation[3]
    child2_x, child2_y = observation[4], observation[5]
    vecna_x, vecna_y = observation[6], observation[7]
    child1_rescued = int(observation[8])
    child2_rescued = int(observation[9])
    proj_x, proj_y = observation[10], observation[11]
    
    # Agent position (coarse)
    discrete_agent_x = int(agent_x / 3.0)
    discrete_agent_y = int(agent_y / 3.0)
    
    # Rescue status (critical state)
    rescue_status = child1_rescued * 2 + child2_rescued  # 0,1,2,3
    
    # Determine current target
    if not child1_rescued and not child2_rescued:
        # Go for nearest child
        dist1 = np.linalg.norm([child1_x - agent_x, child1_y - agent_y])
        dist2 = np.linalg.norm([child2_x - agent_x, child2_y - agent_y])
        if dist1 < dist2:
            target_x, target_y = child1_x, child1_y
        else:
            target_x, target_y = child2_x, child2_y
    elif not child1_rescued:
        target_x, target_y = child1_x, child1_y
    elif not child2_rescued:
        target_x, target_y = child2_x, child2_y
    else:
        target_x, target_y = agent_x, agent_y  # Both rescued - stay safe
    
    # Direction to target
    dx = target_x - agent_x
    dy = target_y - agent_y
    dx_bucket = 0 if abs(dx) < 2 else (1 if dx > 0 else -1)
    dy_bucket = 0 if abs(dy) < 2 else (1 if dy > 0 else -1)
    
    # Vecna threat
    vecna_dist = np.linalg.norm([vecna_x - agent_x, vecna_y - agent_y])
    vecna_threat = 0 if vecna_dist > 5 else (1 if vecna_dist > 3 else 2)
    
    # Projectile threat
    if proj_x > 0 and proj_y > 0:  # Active projectile
        proj_dist = np.linalg.norm([proj_x - agent_x, proj_y - agent_y])
        proj_threat = 0 if proj_dist > 4 else (1 if proj_dist > 2 else 2)
    else:
        proj_threat = 0
    
    return (discrete_agent_x, discrete_agent_y, rescue_status, 
            dx_bucket, dy_bucket, vecna_threat, proj_threat)


def create_universe3_agent():
    """Improved agent for Universe 3: Strategic dual rescue (6 Demogorgons + Vecna)"""
    
    Q3 = {}
    
    # Hyperparameters for complex multi-objective task
    LEARNING_RATE = 0.45
    DISCOUNT_FACTOR = 0.96
    exploration = {'epsilon': 0.35, 'decay': 0.998, 'min': 0.06}
    
    def select_action(observation):
        state = discretize_universe3(observation)
        
        if random.random() < exploration['epsilon']:
            # Smart exploration toward current target
            agent_x, agent_y = observation[0], observation[1]
            child1_x, child1_y = observation[2], observation[3]
            child2_x, child2_y = observation[4], observation[5]
            child1_rescued = observation[8]
            child2_rescued = observation[9]
            
            # Determine target
            if not child1_rescued and not child2_rescued:
                dist1 = np.linalg.norm([child1_x - agent_x, child1_y - agent_y])
                dist2 = np.linalg.norm([child2_x - agent_x, child2_y - agent_y])
                target_x = child1_x if dist1 < dist2 else child2_x
                target_y = child1_y if dist1 < dist2 else child2_y
            elif not child1_rescued:
                target_x, target_y = child1_x, child1_y
            elif not child2_rescued:
                target_x, target_y = child2_x, child2_y
            else:
                return random.randint(0, 3)  # Both rescued
            
            dx = target_x - agent_x
            dy = target_y - agent_y
            
            # 70% bias toward target
            if random.random() < 0.7:
                if abs(dx) > abs(dy):
                    return 0 if dx > 0 else 2
                else:
                    return 1 if dy > 0 else 3
            else:
                return random.randint(0, 3)
        
        q_values = [Q3.get((state, a), 0.0) for a in range(4)]
        max_q = max(q_values)
        best_actions = [a for a in range(4) if q_values[a] == max_q]
        return random.choice(best_actions)
    
    def update(state, action_index, reward, next_state):
        old_q = Q3.get((state, action_index), 0.0)
        future_q_values = [Q3.get((next_state, a), 0.0) for a in range(4)]
        max_future_q = max(future_q_values)
        
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q)
        Q3[(state, action_index)] = new_q
        
        exploration['epsilon'] *= exploration['decay']
        exploration['epsilon'] = max(exploration['min'], exploration['epsilon'])
    
    return select_action, update, Q3


# ====================================================================
# MAIN INTERFACE (REQUIRED BY COMPETITION RUNNER)
# ====================================================================

def get_agent_for_universe(universe_name):
    """
    Return the appropriate agent for each universe.
    
    This is called by competition_runner.py automatically.
    """
    if "FIND WILL" in universe_name:
        return create_universe1_agent(), discretize_universe1
    elif "ESCAPE ROOM" in universe_name or "HAWKINS" in universe_name:
        return create_universe2_agent(), discretize_universe2
    elif "VECNA" in universe_name or "FIGHT" in universe_name:
        return create_universe3_agent(), discretize_universe3
    else:
        raise ValueError(f"Unknown universe: {universe_name}")


def get_agent_info():
    """Return agent configuration"""
    return {
        'team_name': 'Elite AI Squad',
        'agent_type': 'Multi-Agent Q-Learning (Optimized)',
        'learning_rate_u1': 0.5,
        'learning_rate_u2': 0.55,
        'learning_rate_u3': 0.45,
        'discount_factor': '0.93-0.96',
        'exploration_strategy': 'Smart epsilon-greedy (directional bias)',
        'description': 'Improved agents with better discretization and exploration'
    }


# ====================================================================
# TESTING
# ====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IMPROVED MULTI-AGENT SOLUTION")
    print("=" * 70)
    
    print("\n✅ Universe 1: Rescue Will Agent (Smart Pathfinding)")
    obs1 = np.array([5.0, 5.0, 17.0, 15.0, 10.0, 10.0])
    state1 = discretize_universe1(obs1)
    print(f"  Observation: {obs1}")
    print(f"  State: {state1}")
    (select1, update1, Q1), disc1 = get_agent_for_universe("RESCUE WILL")
    action1 = select1(obs1)
    print(f"  Action selected: {action1}")
    
    print("\n✅ Universe 2: Escape Room Agent (Smart Navigation)")
    obs2 = np.array([2.0, 18.0, 17.0, 2.0, 10.0, 10.0])
    state2 = discretize_universe2(obs2)
    print(f"  Observation: {obs2}")
    print(f"  State: {state2}")
    (select2, update2, Q2), disc2 = get_agent_for_universe("ESCAPE ROOM")
    action2 = select2(obs2)
    print(f"  Action selected: {action2}")
    
    print("\n✅ Universe 3: Defeat Vecna Agent (Strategic Rescue)")
    obs3 = np.array([10.0, 2.0, 3.0, 10.0, 17.0, 10.0, 
                     10.0, 10.0, 0.0, 0.0, 12.0, 8.0])
    state3 = discretize_universe3(obs3)
    print(f"  Observation: {obs3}")
    print(f"  State: {state3}")
    (select3, update3, Q3), disc3 = get_agent_for_universe("VECNA")
    action3 = select3(obs3)
    print(f"  Action selected: {action3}")
    
    print("\n" + "=" * 70)
    print("✅ THREE IMPROVED AGENTS INITIALIZED!")
    print("=" * 70)
    print("\nKey Improvements:")
    print("  1. Smarter discretization (fewer states, better features)")
    print("  2. Directional exploration bias (faster learning)")
    print("  3. Higher learning rates (faster convergence)")
    print("  4. Better threat awareness (safety + goal balance)")
    print("\nAgent Info:")
    for k, v in get_agent_info().items():
        print(f"  {k}: {v}")