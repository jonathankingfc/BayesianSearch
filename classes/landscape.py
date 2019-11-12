import numpy as np


class Landscape:
    """
    This is a class for representing our Landscape environment.

    Attributes:
        grid: The Landscape represented as a 2-D array. Each location will contain a string reprenting the terrain of the cell. 
        target: The location of the target as a tuple. 
        dim: The dimension of the environment. The number of cells in the landscape will be (dim x dim).
        false_negative_probabilities: The probabilites for a false negative for each terrain type.
    """

    def __init__(self, dim):
        """
        Initialize Landscape and set attributes.
        """

        # Set attributes
        self.dim = dim

        # P(Not found | Target Present) for each cell type:
        self.false_negative_probabilities = {
            "flat": 0.1,
            "hilly": 0.3,
            "forest": 0.7,
            "caves": 0.9
        }

        # Create grid
        terrains = ['flat', 'hilly', 'forest', 'caves']

        self.grid = np.random.choice(
            terrains, size=(dim, dim), p=[0.2, 0.3, 0.3, 0.2])

        # Set target location
        self.target = (np.random.randint(dim), np.random.randint(dim))

    def resolve_query(self, query_loc):
        """
        Resolve a query from the belief agent.
        """

        # If the query location is located at the target
        if query_loc == self.target:

            # Get query location type
            query_loc_type = self.grid[query_loc[0], query_loc[1]]

            # Get probabilities of finding and not finding
            prob_not_found = self.false_negative_probabilities[query_loc_type]
            prob_found = 1 - prob_not_found

            return np.random.choice([False, True], p=[
                prob_not_found, prob_found])

        # Otherwise, return false
        else:
            return False

    def pick_new_target(self):
        """
        Sets target to new location
        """

        # Set target location
        self.target = (np.random.randint(self.dim),
                       np.random.randint(self.dim))

    def pprint(self):
        """
        Print landscape in easier format. 
        """

        print('\n'.join([''.join(['{:10}'.format(item) for item in row])
                         for row in self.grid]))
