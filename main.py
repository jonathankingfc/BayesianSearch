# Import packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns

# Import classes
from classes.landscape import Landscape
from classes.belief import BeliefGrid


def main():

    search(2)


def search(rule):
    """
    Create a landscape and perform a search for the target. The search will require a rule to decide which cells to search.
    """

    # Create landscape
    landscape = Landscape(20)

    # Create belief state and set initial belief probabilities
    belief_state = BeliefGrid(landscape)

    prob_of_finding_grid_states = []
    prob_target_present_grid_states = []

    # Keep running the process until target if found
    while True:

        print(belief_state.t)

        # Choose query location based on rule
        query_loc = belief_state.choose_query_loc(rule)

        # Resolve query
        result = landscape.resolve_query(query_loc)

        # If target was found
        if result == True:
            break

        # Otherwise update belief grid
        else:

            # Update beliefs
            belief_state.update_prob_present(query_loc)
            belief_state.update_prob_finding()
            belief_state.t += 1

            # Append new belief states to output : THIS IS USED FOR PLOTTING. NOT CRUCIAL TO ALGORITHM
            prob_of_finding_grid_states.append(
                belief_state.get_prob_finding_grid())

            prob_target_present_grid_states.append(
                belief_state.get_prob_present_grid())

    draw_heatmap(prob_of_finding_grid_states, landscape.dim)
    # [print(map) for map in prob_target_present_grid_states]


def draw_heatmap(belief_states, dim):

    fig = plt.figure()

    def init():
        plt.clf()
        sns.heatmap(np.zeros((dim, dim)), square=True)

    def animate(state):
        plt.clf()
        sns.heatmap(state,  square=True)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=belief_states, repeat=True, interval=1)

    plt.show()


if __name__ == "__main__":
    main()
