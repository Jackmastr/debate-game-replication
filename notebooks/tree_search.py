import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import time
import math
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to logging.DEBUG for detailed debug output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Action -> Node
        self.visit_count_n = 0
        self.value_sum_w = 0
        self.mean_value_q = 0
        legal_actions = state.get_legal_actions()
        self.prior_p = 1/len(legal_actions) if legal_actions else 0

    def update(self, value):
        self.visit_count_n += 1
        self.value_sum_w += value
        self.mean_value_q = self.value_sum_w / self.visit_count_n

    def get_root(self):
        node = self
        while node.parent:
            node = node.parent
        return node

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, judge, num_rollouts=10_000, cpuct=1):
        self.judge = judge
        self.num_rollouts = num_rollouts
        self.cpuct = cpuct

    def search(self, root_state):
        self.root = Node(root_state)
        for rollout in range(1, self.num_rollouts + 1):
            if rollout % 1000 == 0 or rollout == 1:
                logging.debug(f"Rollout {rollout}/{self.num_rollouts}")
            cur_node = self.root
            cur_state = root_state
            # Selection phase
            while not cur_node.is_leaf() and not cur_state.is_terminal():
                action, cur_node = self.select_best_action(cur_node, cur_state)
                cur_state = cur_state.apply_action(action)
            # Expansion and Simulation
            if not cur_state.is_terminal():
                self.expand(cur_node, cur_state)
            reward = self.simulate(cur_state)
            # Backpropagation
            self.backpropagate(cur_node, reward)

    def select_best_action(self, node, state):
        actions = list(node.children.keys())
        scores = [
            child.mean_value_q + self.cpuct * child.prior_p *
            math.sqrt(node.visit_count_n) / (1 + child.visit_count_n)
            for child in node.children.values()
        ]
        best_score = max(scores)
        best_actions = [a for a, s in zip(actions, scores) if s == best_score]
        chosen_action = random.choice(best_actions)
        return chosen_action, node.children[chosen_action]

    def expand(self, node, state):
        """
        Expand the node by adding all possible child nodes.
        """
        if node.is_leaf() and not state.is_terminal():
            legal_actions = state.get_legal_actions()
            logging.debug(
                f"Expanding node with legal actions: {legal_actions}")
            for action in legal_actions:
                next_state = state.apply_action(action)
                node.children[action] = Node(next_state, parent=node)
        return node

    def simulate(self, state):
        """
        Perform a simulation (rollout) from the given state to a terminal state.
        Returns the reward for the simulation.
        """
        current_state = state  # state.clone()
        while not current_state.is_terminal():
            legal_actions = current_state.get_legal_actions()
            if not legal_actions:
                break
            # Randomly select an action (row, col)
            action = random.choice(legal_actions)
            # logging.debug(f"Simulation: Taking action {action}")
            current_state = current_state.apply_action(action)
        reward = self.judge.evaluate(current_state)
        # logging.debug(f"Simulation reward: {reward}")
        return reward

    def backpropagate(self, node, reward):
        """
        Propagate the simulation result up the tree, updating visit counts and value sums.
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self, node):
        """
        Select the action with the highest visit count from the root node.
        """
        best_visits = -1
        best_action = None
        for action, child in node.children.items():
            if child.visit_count_n > best_visits:
                best_visits = child.visit_count_n
                best_action = action
        return best_action


class MCTSRandom(MCTS):
    def __init__(self, judge, num_rollouts=10_000, cpuct=1):
        super().__init__(judge, num_rollouts, cpuct)

    def select(self, node, state):
        while not node.is_leaf() and not state.is_terminal():
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break
            # Randomly select an action (row, col)
            action = random.choice(legal_actions)
            chosen_child = node.children[action]
            node = chosen_child
        return node


class DebateState:
    def __init__(self, image, label, num_pixels_to_reveal=6):
        """
        Initialize the DebateState with the given image, label, and number of pixels to reveal.

        Args:
            image (torch.Tensor): The true image tensor.
            label (int): The true label of the image.
            num_pixels_to_reveal (int, optional): The total number of pixels to reveal in the game. Defaults to 6.
        """
        self.image = image
        self.label = label
        self.mask = torch.zeros_like(image)
        self.num_pixels_to_reveal = num_pixels_to_reveal
        self.current_player = 0  # Player 0 starts
        self.num_pixels_revealed = 0
        self.unrevealed_pixels = set(self.get_nonzero_pixels())

    @lru_cache(maxsize=None)
    def get_nonzero_pixels(self):
        return tuple(map(tuple, (self.image != 0).nonzero(as_tuple=False).tolist()))

    # def get_nonzero_pixels(self):
    #     """
    #     Get the 2D coordinates of all nonzero pixels in the image.

    #     Returns:
    #         List[Tuple[int, int]]: A list of (row, col) tuples representing nonzero pixels.
    #     """
    #     nonzero_indices = (self.image != 0).nonzero(as_tuple=False)
    #     return [tuple(idx.tolist()) for idx in nonzero_indices]

    def get_unrevealed_pixels(self):
        """
        Get the 2D coordinates of all unrevealed pixels in the image.

        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing unrevealed pixels.
        """
        return list(self.unrevealed_pixels)

    @lru_cache(maxsize=None)
    def is_terminal(self):
        """
        Check if the game has reached a terminal state.

        Returns:
            bool: True if the number of revealed pixels meets or exceeds the limit, False otherwise.
        """
        return self.num_pixels_revealed >= self.num_pixels_to_reveal

    @lru_cache(maxsize=None)
    def get_legal_actions(self):
        """
        Get all legal actions. A legal action is a pixel that is (1) nonzero and (2) not yet revealed and (3) the game is not in a terminal state.

        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing unrevealed pixels.
        """
        if self.is_terminal():
            return []
        return list(self.unrevealed_pixels)

    def apply_action(self, action):
        """
        Apply an action by revealing the specified pixel.

        Args:
            action (Tuple[int, int]): The (row, col) coordinates of the pixel to reveal.

        Returns:
            DebateState: A new state with the action applied.
        """
        new_state = DebateState(
            image=self.image,
            label=self.label,
            num_pixels_to_reveal=self.num_pixels_to_reveal
        )
        new_state.mask = self.mask.clone()
        new_state.mask[action] = 1
        new_state.current_player = 1 - self.current_player
        new_state.num_pixels_revealed = self.num_pixels_revealed + 1
        new_state.unrevealed_pixels = self.unrevealed_pixels - {action}
        # new_state.unrevealed_pixels = self.unrevealed_pixels.copy()
        # new_state.unrevealed_pixels.remove(action)
        return new_state

    def clone(self):
        """
        Create a deep copy of the current state.

        Returns:
            DebateState: A cloned copy of the current state.
        """
        return DebateState(
            image=self.image.clone(),
            label=self.label,
            num_pixels_to_reveal=self.num_pixels_to_reveal
        )._copy_attributes(self)

    def _copy_attributes(self, other):
        """
        Helper method to copy attributes from another state.

        Args:
            other (DebateState): The state to copy attributes from.

        Returns:
            DebateState: The current state with copied attributes.
        """
        self.mask = other.mask.clone()
        self.current_player = other.current_player
        self.num_pixels_revealed = other.num_pixels_revealed
        return self

    def get_current_player(self):
        """
        Get the current player.

        Returns:
            int: The current player's identifier (0 or 1).
        """
        return self.current_player

    def __repr__(self):
        """
        Return a string representation of the DebateState.

        Returns:
            str: String representation of the state.
        """
        return (f"DebateState(current_player={self.current_player}, "
                f"pixels_revealed={torch.sum(self.mask).item()}/{self.num_pixels_to_reveal})")

    def get_observation(self):
        """
        Generate the observation by stacking the mask and masked image along the channel dimension.

        Returns:
            torch.Tensor: A tensor of shape (1, 2, 28, 28) representing the mask and masked image.
        """
        masked_image = self.image * self.mask  # Shape: [28, 28]
        masked_image = masked_image.unsqueeze(0)  # Shape: [1, 28, 28]
        # Shape: [1, 2, 28, 28]
        stacked = torch.stack([self.mask.unsqueeze(0), masked_image], dim=1)
        # Ensure tensor is on the correct device
        return stacked.to(self.image.device)


class Judge:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Ensure the model is in evaluation mode

    @torch.no_grad()
    def evaluate(self, state):
        """
        Evaluate the state using the model.
        Args:
            state (DebateState): The current state of the game.

        Returns:
            float: The evaluation score of the state.
        """
        observation = state.get_observation()
        output = self.model(observation)
        probabilities = F.softmax(output, dim=1)
        true_label_prob = probabilities[0, state.label].item()
        return true_label_prob if state.current_player == 0 else 1 - true_label_prob

    def predict(self, state):
        """
        Predict the label based on the model's highest probability.

        Args:
            state (DebateState): The current state of the game.

        Returns:
            int: The predicted label with the highest probability.
        """
        observation = state.get_observation()
        with torch.no_grad():
            output = self.model(observation)
        probabilities = F.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        return predicted_label
