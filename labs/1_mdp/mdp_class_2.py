import numpy as np
import time
from mdf_definitions import S, A, P, gamma, Rw

class MarkovDecisionProblemV2:
    """
    A class to represent and solve a Markov Decision Problem (MDP).

    Attributes:
        S (list): A list of state names.
        A (list): A list of action names.
        P (np.ndarray): Transition probabilities, shape (|A|, |S|, |S|).
        R (np.ndarray): Reward function, shape (|S|, |A|).
        gamma (float): Discount factor.
        num_states (int): The number of states |S|.
        num_actions (int): The number of actions |A|.
    """

    def __init__(self, states, actions, transitions, rewards, gamma):
        self.S = states
        self.A = actions
        self.P = np.array(transitions)
        self.R = rewards
        self.gamma = gamma
        self.num_states = len(states)
        self.num_actions = len(actions)

        # A dictionary to map state names to their grid coordinates for policy creation
        self.state_coords = {
            '1':(0,0), '2':(0,1), '3':(0,2),
            '4':(1,0), '5':(1,1), '6':(1,2),
            '7':(2,1)
        }
        self.goal_coord = self.state_coords['6']

    def _get_q_values(self, V: np.ndarray) -> np.ndarray:
        """Helper to calculate Q-values for a given state-value function V."""
        return self.R + self.gamma * self.P.dot(V).T

    def evaluate_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Performs Policy Evaluation to find the state-value function V^π.
        Solves the Bellman equation for V^π: V^π = (I - γP^π)^-1 * R^π.
        """
        R_pi = (policy * self.R).sum(axis=1)
        P_pi = (policy[:, :, np.newaxis] * self.P.transpose(1, 0, 2)).sum(axis=1)
        I = np.eye(self.num_states)
        V = np.linalg.solve(I - self.gamma * P_pi, R_pi)
        return V

    def value_iteration(self, tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray, int, float]:
        """
        Finds the optimal policy and value function using Value Iteration.
        """
        start_time = time.time()
        V = np.zeros(self.num_states)
        iterations = 0
        while True:
            V_old = V.copy()
            Q = self._get_q_values(V_old)
            V = Q.max(axis=1)
            err = np.linalg.norm(V - V_old)
            iterations += 1
            if err < tolerance:
                break

        optimal_policy = self._extract_policy_from_q(Q)
        elapsed_time = time.time() - start_time
        return V, optimal_policy, iterations, elapsed_time

    def policy_iteration(self, initial_policy: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
        """
        Finds the optimal policy and value function using Policy Iteration.
        """
        start_time = time.time()
        policy = initial_policy.copy()
        iterations = 0
        while True:
            old_policy = policy.copy()
            V = self.evaluate_policy(policy)
            Q = self._get_q_values(V)
            policy = self._extract_policy_from_q(Q)
            iterations += 1
            if np.array_equal(policy, old_policy):
                break

        elapsed_time = time.time() - start_time
        return V, policy, iterations, elapsed_time

    def _extract_policy_from_q(self, q_matrix: np.ndarray) -> np.ndarray:
        """Extracts a deterministic policy from Q-values."""
        best_actions = np.isclose(q_matrix, q_matrix.max(axis=1, keepdims=True), atol=1e-8, rtol=1e-8)
        return best_actions / best_actions.sum(axis=1, keepdims=True)

    def create_closest_to_goal_policy(self) -> np.ndarray:
        """Creates the policy described in Activity 2."""
        policy = np.zeros((self.num_states, self.num_actions))

        for s_idx, state_name in enumerate(self.S):
            cell_num = state_name[0]

            if cell_num == '6': # Goal state
                policy[s_idx, :] = 0.25 # Any action is fine
                continue

            current_coord = self.state_coords[cell_num]

            # Calculate distances for each action
            distances = []
            possible_moves = {
                'U': (current_coord[0] - 1, current_coord[1]),
                'D': (current_coord[0] + 1, current_coord[1]),
                'L': (current_coord[0], current_coord[1] - 1),
                'R': (current_coord[0], current_coord[1] + 1)
            }

            for action in self.A:
                move_coord = possible_moves[action]
                # Manhattan distance
                dist = abs(move_coord[0] - self.goal_coord[0]) + abs(move_coord[1] - self.goal_coord[1])
                distances.append(dist)

            # Find the minimum distance and assign probabilities
            min_dist = np.min(distances)
            best_actions = np.where(np.isclose(distances, min_dist))[0]
            prob = 1.0 / len(best_actions)
            for a_idx in best_actions:
                policy[s_idx, a_idx] = prob

        return policy

    def simulate_trajectories(self, initial_state_idx: int, policy: np.ndarray, num_traj: int, steps: int):
        """Simulates trajectories following a given policy."""
        total_rewards = []
        for _ in range(num_traj):
            current_state = initial_state_idx
            discounted_reward = 0.0
            for t in range(steps):
                action_probs = policy[current_state]
                chosen_action = np.random.choice(self.num_actions, p=action_probs)

                reward = self.R[current_state, chosen_action]
                discounted_reward += (self.gamma ** t) * reward

                transition_probs = self.P[chosen_action, current_state, :]
                current_state = np.random.choice(self.num_states, p=transition_probs)

            total_rewards.append(discounted_reward)

        return np.mean(total_rewards)

mdp = MarkovDecisionProblemV2(S, A, P, Rw, gamma)
print("--- MDP Model Initialized ---")
print(f"Number of States: {mdp.num_states}")
print(f"Number of Actions: {mdp.num_actions}")

# --- Activity 2: Describe the Policy ---
# This policy moves the agent to the cell closest to the goal.
# If multiple cells are equally close, it chooses randomly between them.
pi_closest = mdp.create_closest_to_goal_policy()
print("\n--- Policy Matrix (Closest to Goal) ---")
print(pi_closest)

# Activity 3: Compute State-Value Function ($V^\\pi$)
# Calculate Vπ for the policy from Activity 2
V_pi_closest = mdp.evaluate_policy(pi_closest)

print("--- State-Value Function (Vπ) for 'Closest to Goal' Policy ---")
print(V_pi_closest.reshape(-1, 1))

# Activity 4: Control using Value Interation
# Compute V* using Value Iteration
V_star_vi, pi_star_vi, iterations_vi, time_vi = mdp.value_iteration(tolerance=1e-8)

print("\n--- Value Iteration Results ---")
print(f"Algorithm converged in {iterations_vi} iterations.")
print(f"Time taken: {time_vi:.4f} seconds.")
print("\nOptimal Value Function (V*):")
print(V_star_vi)

# Compare V* with Vπ from Activity 3
are_functions_equal = np.allclose(V_pi_closest, V_star_vi)
print("\n--- Comparison: V* vs. Vπ (Closest to Goal) ---")
if not are_functions_equal:
    print("✅ As expected, V* is different from Vπ.")
    print("This confirms the 'closest to goal' policy is not optimal.")
else:
    print("V* is the same as Vπ, which is unexpected.")


# Activity 5: Control using Policy Iteration
# Next, we'll compute the optimal policy and value function again,
# this time using the Policy Iteration algorithm.
# We will compare its efficiency (iterations and time) with that of Value Iteration.

# Create an initial uniform policy to start the algorithm
initial_policy = np.full((mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions)

# Compute π* and V* using Policy Iteration
V_star_pi, pi_star_pi, iterations_pi, time_pi = mdp.policy_iteration(initial_policy)

print("\n--- Policy Iteration Results ---")
print(f"Algorithm converged in {iterations_pi} iterations.")
print(f"Time taken: {time_pi:.4f} seconds.")
print("\nOptimal Policy (π*):")
# We use argmax to show the deterministic action for each state
print(np.argmax(pi_star_pi, axis=1))
print("\nOptimal Value Function (V*):")
print(V_star_pi)

print("\n--- Algorithm Comparison ---")
print(f"Value Iteration:   {iterations_vi} iterations in {time_vi:.4f} seconds.")
print(f"Policy Iteration:  {iterations_pi} iterations in {time_pi:.4f} seconds.")

# Verify that both algorithms found the same optimal value function
are_v_star_equal = np.allclose(V_star_vi, V_star_pi)
print(f"\nValue functions from both algorithms are equal: {are_v_star_equal}")

# Activity 6: Simulation
# Finally:
# we simulate an agent following the optimal policy found above to empirically verify our theoretical calculations.
# For this simulation:
# we'll test three scenarios starting from the physical location of cell 2,
# as it offers states with different key combinations
# ("no keys", "red key only", and "both keys").

# Define the starting states for each scenario based on cell 2
# (i) no keys -> state '2' (index 1)
# (ii) only red key -> state '2R' (index 2)
# (iii) both keys -> state '2BR' (index 3)
start_states = {
    "No Keys (start at '2')": 1,
    "Red Key Only (start at '2R')": 2,
    "Both Keys (start at '2BR')": 3
}

print("--- Simulating Trajectories (100 runs, 10,000 steps each) ---")

for scenario, start_idx in start_states.items():
    # Simulate 100 trajectories
    avg_reward = mdp.simulate_trajectories(
        initial_state_idx=start_idx,
        policy=pi_star_pi,
        num_traj=100,
        steps=10000
    )

    # Compare with the theoretical value from V*
    theoretical_value = V_star_pi[start_idx]

    print(f"\nScenario: {scenario}")
    print(f"  - Simulated Average Reward: {avg_reward:.4f}")
    print(f"  - Theoretical V* Value:     {theoretical_value:.4f}")