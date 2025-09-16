import time
import numpy as np
from mdf_definitions import S, A, U, D, L, R, gamma, Rw

class MarkovDecisionProblemLabActivity:
  """
  Classe utilitária para representar um processo de Decisão de Markov (MDP)
  - S - states = Conjunto Discreto de Estados
  - A - actions = Conjunto Discreto de Ações
  - P(s'| s, a) = Modelo de Transição Estocástico
  - R(s, a) = Função de Recompensa
  - gamma = Taxa de Desconto Gamma
  """

  def __init__(
    self,
    states: list[str],
    actions: list[str],
    up_transitions: np.ndarray,
    down_transitions: np.ndarray,
    left_transitions: np.ndarray,
    right_transitions: np.ndarray,
    reward_function: np.ndarray,
    discount_rate: float
  ) -> None:
    self.S = states
    self.num_states = len(states)
    self.A = actions
    self.num_actions = len(actions)
    self.P = [up_transitions, down_transitions, left_transitions, right_transitions]
    self.RW = reward_function
    self.gamma = discount_rate

  def describe_initial_data(self):
    print(f'States[{self.num_states}]:', self.S)
    print(f'Actions[{self.num_actions}]:', self.A)
    for transition, action in zip(self.P, self.A):
      print(f"Transition shape for action '{action}': {transition.shape}")
    print(f'Reward function shape:\n{self.RW.shape}')
    print(f'Discount rate: {self.gamma}')

  def create_initial_policy(self) -> np.ndarray:
    """
    Creates an initial policy where each action has an equal probability.

    Returns:
        A numpy array representing the initial policy. Shape (num_states, num_actions).
    """
    prob_per_action = 1.0 / self.num_actions
    policy_matrix = np.full((self.num_states, self.num_actions), prob_per_action)
    return policy_matrix

  def manual_policy(self) -> np.ndarray:
    """
    Describes the policy that, in each state s, always moves the agent to the cell
    closest to the goal (regardless of the number of keys in the agent's possession).
    If multiple of these cells exist, the agent should select randomly between them.

    Returns:
        A numpy array representing the policy. Shape (num_states, num_actions),
        where policy[s, a] has the probability of selecting action a in state s.
    """
    # S = ['1BR', '2', '2R', '2BR', '3', '3R', '3BR', '4', '4R', '4BR', '5', '5R', '5BR', '6BR', '7R', '7BR']
    # A = ['U', 'D', 'L', 'R']
    policy = np.array([
        # State '1BR' (closest is 4BR or 2BR) - move R or D
        [0.0, 0.5, 0.0, 0.5],
        # State '2' (closest is 4 or 3) - move D or R
        [0.0, 0.5, 0.0, 0.5],
        # State '2R' (closest is 4R or 3R) - move D or R
        [0.0, 0.5, 0.0, 0.5],
        # State '2BR' (closest is 4BR or 3BR) - move D or R
        [0.0, 0.5, 0.0, 0.5],
        # State '3' (closest is 5) - move D
        [0.0, 1.0, 0.0, 0.0],
        # State '3R' (closest is 5R) - move D
        [0.0, 1.0, 0.0, 0.0],
        # State '3BR' (closest is 5BR) - move D
        [0.0, 1.0, 0.0, 0.0],
        # State '4' (closest is 5) - move R
        [0.0, 0.0, 0.0, 1.0],
        # State '4R' (closest is 5R) - move R
        [0.0, 0.0, 0.0, 1.0],
        # State '4BR' (closest is 5BR) - move R
        [0.0, 0.0, 0.0, 1.0],
        # State '5' (closest is 6BR) - move R
        [0.0, 0.0, 0.0, 1.0],
        # State '5R' (closest is 6BR) - move R
        [0.0, 0.0, 0.0, 1.0],
        # State '5BR' (closest is 6BR) - move R
        [0.0, 0.0, 0.0, 1.0],
        # State '6BR' (Goal) - Stay
        [0.25, 0.25, 0.25, 0.25], # Assuming staying in goal is optimal, equally likely to attempt any action
        # State '7R' (closest is 4R) - move U
        [1.0, 0.0, 0.0, 0.0],
        # State '7BR' (closest is 4BR) - move U
        [1.0, 0.0, 0.0, 0.0]
    ])
    # Normalize probabilities to sum to 1 for each state (row)
    policy = policy / np.sum(policy, axis=1, keepdims=True)
    return policy


  def policy_iteration(self) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Executa o algoritmo Policy Iteration para encontrar a política ótima π*.

    Returns:
        Uma tupla contendo a política ótima π*, a função de valor ótima V*,
        o número de iterações e o tempo decorrido.
    """
    print("\n--- Executando o Policy Iteration para encontrar π* ---")
    start_time = time.time()

    # 1. Inicia com uma política aleatória
    policy = self.create_initial_policy()
    policy_stable = False
    iterations = 0
    V = np.zeros((self.num_states, 1))

    # 2. Itera até a política convergir
    while not policy_stable:
        # Armazena a política antiga para comparação
        old_policy = policy.copy()

        # Etapa de Avaliação: Calcula Vπ para a política atual
        V = self.state_value_function(policy)

        # Etapa de Melhora: Extrai uma nova política "gananciosa" com base em Vπ
        Q_values = np.zeros((self.num_states, self.num_actions))
        for a_idx in range(self.num_actions):
            Q_values[:, a_idx] = self.RW[:, a_idx] + self.gamma * (self.P[a_idx] @ V.flatten())

        policy = self.extract_policy_from_q(Q_values)

        # Verifica se a política convergiu
        policy_stable = np.allclose(policy, old_policy)
        iterations += 1

    elapsed_time = time.time() - start_time

    return policy, V.reshape(-1, 1), iterations, elapsed_time


  def state_value_function(self, policy_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the state-value function Vπ for a given policy π using the Bellman equation.

    Args:
        policy_matrix: A numpy array (num_states x num_actions) with the probabilities π(a|s).

    Returns:
        A numpy array (num_states x 1) with the value of each state.
    """
    print("\n--- Calculando a Função de Valor de Estado (Vπ) ---")

    # Rπ(s) = Σ [π(a|s) * R(s,a)] para cada estado s
    R_pi = (policy_matrix * self.RW).sum(axis=1)

    # Pπ(s'|s) = Σ [π(a|s) * P(s'|s,a)] para cada par (s, s')
    P_pi = np.zeros((self.num_states, self.num_states))
    for a_idx, P_a in enumerate(self.P):
        # Pondera a matriz de transição da ação 'a' pela probabilidade de tomá-la
        P_pi += policy_matrix[:, a_idx, np.newaxis] * P_a

    # Solve Vπ = (I - γ * Pπ)^-1 * Rπ
    I = np.eye(self.num_states)
    V_pi = np.linalg.solve(I - self.gamma * P_pi, R_pi)


    return V_pi.reshape(-1, 1) # Return as a column vector


  def value_iteration(self, tolerance: float = 1e-8) -> tuple[np.ndarray, int, float]:
    """
    Executes the Value Iteration algorithm to find the optimal value function V*.

    Args:
        tolerance: The stopping criterion for convergence.

    Returns:
        A tuple containing V*, the number of iterations, and the elapsed time.
    """
    print("\n--- Executando o Value Iteration para encontrar V* ---")
    start_time = time.time()

    V = np.zeros(self.num_states)
    err = float('inf')
    iterations = 0

    while err > tolerance:
        V_old = V.copy()

        Q_values = np.zeros((self.num_states, self.num_actions))

        for a_idx in range(self.num_actions):
            P_a = self.P[a_idx]
            R_a = self.RW[:, a_idx]
            Q_values[:, a_idx] = R_a + self.gamma * (P_a @ V_old)

        V = Q_values.max(axis=1)

        err = np.linalg.norm(V - V_old)
        iterations += 1

    elapsed_time = time.time() - start_time

    return V.reshape(-1, 1), iterations, elapsed_time

  def extract_policy_from_q(self, q_matrix: np.ndarray) -> np.ndarray:
    """
    Extracts a deterministic optimal policy from a matrix of Q-values.
    """
    optimal_actions = np.isclose(q_matrix, q_matrix.max(axis=1, keepdims=True)).astype(float)
    policy = optimal_actions / optimal_actions.sum(axis=1, keepdims=True)
    return policy


  def compare_policies_equality(self, policy_a: np.ndarray, policy_b: np.ndarray, policy_a_name="Policy A", policy_b_name="Policy B") -> bool:
    print(f"\n--- Comparando IGUALDADE: '{policy_a_name}' vs. '{policy_b_name}' ---")
    are_identical = np.allclose(policy_a, policy_b)
    if are_identical:
        print(f"✅ As políticas são idênticas.")
    else:
        print(f"❌ As políticas são diferentes.")
    return are_identical

  def compare_policy_performance(self, policy_a: np.ndarray, policy_b: np.ndarray, policy_a_name="Policy A", policy_b_name="Policy B"):
    """
    Compares the performance of two policies based on their state-value functions (Vπ).

    Args:
        policy_a: The first policy matrix.
        policy_b: The second policy matrix.
        policy_a_name: Name of the first policy for display.
        policy_b_name: Name of the second policy for display.
    """
    print(f"\n--- Comparando DESEMPENHO: '{policy_a_name}' vs. '{policy_b_name}' ---")

    V_a = self.state_value_function(policy_a)
    V_b = self.state_value_function(policy_b)
    avg_a = V_a.mean()
    avg_b = V_b.mean()
    winner = f"'{policy_a_name}' (média: {avg_a:.2f})" if avg_a > avg_b else f"'{policy_b_name}' (média: {avg_b:.2f})"
    loser = f"'{policy_b_name}' (média: {avg_b:.2f})" if avg_a > avg_b else f"'{policy_a_name}' (média: {avg_a:.2f})"

    if np.allclose(V_a, V_b):
        print(f"⚖️ As políticas '{policy_a_name}'  {avg_a:.2f}  e '{policy_b_name}' {avg_b:.2f} têm desempenho equivalente.")
    elif np.all(V_a >= V_b):
        print(f"🏆 A política '{policy_a_name}' {avg_a:.2f} é melhor ou igual à política '{policy_b_name}' {avg_b:.2f}.")
    elif np.all(V_b >= V_a):
        print(f"🏆 A política '{policy_b_name}' {avg_b:.2f} é melhor ou igual à política '{policy_a_name}' {avg_a:.2f}.")
    else:
        print(f"🤔 Nenhuma política é estritamente superior à outra em todos os estados.")
        print(f"   No entanto, a política {winner} possui um valor médio superior a {loser}.")


mdp = MarkovDecisionProblemLabActivity(
    states=['1BR', '2', '2R', '2BR', '3', '3R', '3BR', '4', '4R', '4BR', '5', '5R', '5BR', '6BR', '7R', '7BR'],
    actions=['U', 'D', 'L', 'R'],
    up_transitions=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]]),
    down_transitions=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    left_transitions=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    right_transitions=np.array([[0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    reward_function=np.array([[0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0]]),
    discount_rate=0.99
)

mdp.describe_initial_data()

# Get the initial policy and the policy from Activity 2
initial_policy = mdp.create_initial_policy()
policy_from_activity_2 = mdp.manual_policy()
optimal_policy, V_optimal, pi_iterations, pi_time = mdp.policy_iteration()
V_initial_policy = mdp.state_value_function(initial_policy)
V_policy_activity_2 = mdp.state_value_function(policy_from_activity_2)

mdp.compare_policies_equality(
    initial_policy,
    policy_from_activity_2,
    policy_a_name="Initial Policy",
    policy_b_name="Policy from Activity 2"
)

mdp.compare_policy_performance(
    initial_policy,
    policy_from_activity_2,
    policy_a_name="Initial Policy",
    policy_b_name="Policy from Activity 2"
)


mdp.compare_policy_performance(
    optimal_policy,
    policy_from_activity_2,
    policy_a_name="Optimal Policy",
    policy_b_name="Policy from Activity 2"
)

mdp.compare_policy_performance(
    optimal_policy,
    initial_policy,
    policy_a_name="Optimal Policy",
    policy_b_name="Initial Policy"
)

#agent1: no keys
policy1 =np.array([
    # pos. 1
    [0.0, 0.0, 0.0, 1.0],
    # pos. 2
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.5, 0.0, 0.5],
    # pos. 3
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    # pos. 4
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    # pos. 5
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    # pos. 6
    [1.0/3, 1.0/3, 0.0, 1.0/3],
    # pos. 7
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0]
])
#agent2: red key only
# S = ['1BR', '2', '2R', '2BR', '3', '3R', '3BR', '4', '4R', '4BR', '5', '5R', '5BR', '6BR', '7R', '7BR']
policy2 =np.array([
    # pos. 1
    [0.0, 0.0, 0.0, 1.0],
    # pos. 2
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.5, 0.0, 0.5],
    # pos. 3
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    # pos. 4
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    # pos. 5
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    # pos. 6
    [1.0/3, 1.0/3, 0.0, 1.0/3],
    # pos. 7
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0]
])
#agent3: red and blue key
# S = ['1BR', '2', '2R', '2BR', '3', '3R', '3BR', '4', '4R', '4BR', '5', '5R', '5BR', '6BR', '7R', '7BR']
policy2 =np.array([
    # pos. 1
    [0.0, 0.0, 0.0, 1.0],
    # pos. 2
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    # pos. 3
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    # pos. 4
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    # pos. 5
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    # pos. 6
    [1.0/3, 1.0/3, 0.0, 1.0/3],
    # pos. 7
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0]
])

MDPmodel = MarkovDecisionProblemLabActivity(S, A, U, D, L, R, Rw, gamma)