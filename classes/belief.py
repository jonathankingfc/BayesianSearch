import numpy as np
import random


class BeliefGrid:
    """
    This is a class for representing our solution Agent.

    Attributes:
        grid (np.ndarray): The Agent's knowledge of the landscape.
        dim: The dimension of the environment. The number of cells in the landscape will be (dim x dim).
        t: The time interval t that passes along with each new observation.
        num_actions: The number of actions that the belief grid has made. 1 for search query and 1 for one cell of travel
        false_negative_probabilities: The probabilites for a false negative for each terrain type.
        current_loc: The current location in the belief grid.
    """

    def __init__(self, landscape):
        """
        Initialize belief grid and set initial belief probabilities.
        """

        # Set attributes
        self.t = 0
        self.num_actions = 0
        self.dim = landscape.dim
        self.current_loc = (0, 0)

        # P(Not found | Target Present) for each cell type:
        self.false_negative_probabilities = {
            "flat": 0.1,
            "hilly": 0.3,
            "forest": 0.7,
            "caves": 0.9
        }

        # Create grid
        self.grid = np.empty(shape=(self.dim, self.dim), dtype=object)

        # Create grid of belief cells
        for i in range(self.dim):
            for j in range(self.dim):

                # Get type
                type = landscape.grid[i, j]

                # Create belief cell with type
                initial_belief_cell = BeliefCell(type)

                # Set initial belief of target being present
                initial_belief_cell.prob_target_present = 1/(self.dim**2)

                # Set initial probability of finding the target at this location
                initial_belief_cell.prob_of_finding = (
                    1 - self.false_negative_probabilities[type]) * initial_belief_cell.prob_target_present

                self.grid[i, j] = initial_belief_cell

    def choose_query_loc(self, rule, consider_travel=False, target_not_in_terrain=None):
        """
        Choose the query location based on the maximum probability for our given rule. 
        Rule 1: This will return the cell location with the highest probability of the target being present.
        Rule 2: This will return the cell location with the highest probability of finding the target.
        """

        # Initial maximum location
        query_loc = self.current_loc

        # Iterate through cells to find maximum
        for i in range(self.dim):
            for j in range(self.dim):

                if self.grid[i, j].type == target_not_in_terrain:
                    continue

                # If we are considering travel, update heuristic
                if consider_travel == False:
                    heuristic_curr = 0
                    heuristic_new = 0
                else:
                    heuristic_curr = self.get_manhattan_distance(
                        self.current_loc, query_loc)
                    heuristic_new = self.get_manhattan_distance(
                        self.current_loc, (i, j))

                # Check for rule 1
                if rule == 1:

                    # If probability is greater than current, update query location
                    if self.grid[query_loc[0], query_loc[1]].prob_target_present / (1 + heuristic_curr) < self.grid[i, j].prob_target_present / (1 + heuristic_new):
                        query_loc = (i, j)

                # Check for rule 2
                if rule == 2:

                    # If probability is greater than current, update query location
                    if self.grid[query_loc[0], query_loc[1]].prob_of_finding / (1 + heuristic_curr) < self.grid[i, j].prob_of_finding / (1 + heuristic_new):
                        query_loc = (i, j)

        # Update number of actions
        self.num_actions += self.get_manhattan_distance(
            self.current_loc, query_loc)

        return query_loc

    def update_prob_present(self, query_loc):
        """
        Update probability of target being present in failed query location
        p' = (p * (1 - q)) / (1 - (p * q))
        r' = r / (1 - (p * q))
        where:
            - p is probability of the query cell containing the target, 
            - q is the probability of finding the target given it is present
            - r is the probability of each other cell containing the target
        """

        # Get query location type
        query_loc_type = self.grid[query_loc[0], query_loc[1]].type

        # Get p and q values
        p = self.grid[query_loc[0], query_loc[1]].prob_target_present
        q = 1 - self.false_negative_probabilities[query_loc_type]

        # Update p for failed query location
        self.grid[query_loc[0], query_loc[1]
                  ].prob_target_present = (p * (1 - q))/(1-(p*q))

        # Update r for every other cell
        for i in range(self.dim):
            for j in range(self.dim):

                # Skip over failed query location
                if i != query_loc[0] or j != query_loc[1]:

                    r = self.grid[i, j].prob_target_present
                    self.grid[i, j].prob_target_present = r / (1-(p*q))

    def update_prob_finding(self):
        """
        Update the probability of the target being found for each cell.
        """

        # Iterate through all cells
        for i in range(self.dim):
            for j in range(self.dim):

                cell = self.grid[i, j]
                type = cell.type

                cell.prob_of_finding = (
                    1 - self.false_negative_probabilities[type]) * cell.prob_target_present

    def get_prob_present_grid(self):
        """
        This will return a grid of probabilities for the target being present in each cell.
        """

        # Create empty grid
        probabilites = np.empty(shape=(self.dim, self.dim))

        # Get probabilites from grid
        for i in range(self.dim):
            for j in range(self.dim):
                probabilites[i, j] = self.grid[i, j].prob_target_present

        return probabilites

    def get_prob_finding_grid(self):
        """
        This will return a grid of probabilities for the target being found in each cell.
        """

        # Create empty grid
        probabilites = np.empty(shape=(self.dim, self.dim))

        # Get probabilites from grid
        for i in range(self.dim):
            for j in range(self.dim):
                probabilites[i, j] = self.grid[i, j].prob_of_finding

        return probabilites

    def get_manhattan_distance(self, curr_loc, query_loc):
        """
        Get Manhattan distance from current location to new location. This is the utility of travel.
        """

        return abs(query_loc[0]-curr_loc[0]) + abs(query_loc[1] - curr_loc[1])


class BeliefCell:
    """
    Cell representing the belief probabilities of an individual landscape location.

    Attributes: 
        - prob_of_finding (float): The probability of the agent finding the target in this cell.
        - prob_target_present (float): The true probability of the target being present in this cell.
        - type (string): The 
    """

    def __init__(self, type):
        """
        Initialize belief cell and set attributes
        """

        # Set attributes
        self.type = type
        self.prob_of_finding = 0
        self.prob_target_present = 0
