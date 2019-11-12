# Import packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
import seaborn as sns
from tqdm import tqdm

# Import classes
from classes.landscape import Landscape
from classes.belief import BeliefGrid


def main():

    compare_rules(5)
    compare_rules_with_travel(5)


def compare_rules_with_travel(landscape_dim):
    """
    Compare Rule 1 and Rule 2 given a fixed landscape. This will also take the heuristic into consideration.
    """
    # Create landscape
    landscape = Landscape(landscape_dim)

    # Create space for utility when using heuristic vs not using it
    rule_1_num_actions_without_heuristic = []
    rule_1_num_actions_with_heuristic = []

    # Create space for utility when using heuristic vs not using it
    rule_2_num_actions_without_heuristic = []
    rule_2_num_actions_with_heuristic = []

    # Compute observations for rule 1
    for i in tqdm(range(30)):

        # Find target without using heuristic
        result = search(landscape, 1, consider_travel=False)
        rule_1_num_actions_without_heuristic.append(result['num_actions'])

        # Find target with heuristic
        result = search(landscape, 1, consider_travel=True)
        rule_1_num_actions_with_heuristic.append(result['num_actions'])

        # Reset target location
        landscape.pick_new_target()

    # Compute observations for rule 2
    for i in tqdm(range(30)):

        # Find target without using heuristic
        result = search(landscape, 2, consider_travel=False)
        rule_2_num_actions_without_heuristic.append(result['num_actions'])

        # Find target with heuristic
        result = search(landscape, 2, consider_travel=True)
        rule_2_num_actions_with_heuristic.append(result['num_actions'])

        # Reset target location
        landscape.pick_new_target()

    # Get average observations for each rule with and without using the heuristic
    rule_1_average_without_heuristic = sum(
        rule_1_num_actions_without_heuristic)/len(rule_1_num_actions_without_heuristic)
    rule_2_average_without_heuristic = sum(
        rule_2_num_actions_without_heuristic)/len(rule_2_num_actions_without_heuristic)

    # Get average observations for each rule with and without using the heuristic
    rule_1_average_with_heuristic = sum(
        rule_1_num_actions_with_heuristic)/len(rule_1_num_actions_with_heuristic)
    rule_2_average_with_heuristic = sum(
        rule_2_num_actions_with_heuristic)/len(rule_2_num_actions_with_heuristic)

    print("Rule 1 Average Utility w/o Travel Consideration: " +
          str(rule_1_average_without_heuristic))
    print("Rule 1 Average Utility w Travel Consideration: " +
          str(rule_1_average_with_heuristic))

    print("Rule 2 Average Utility w/o Travel Consideration: " +
          str(rule_2_average_without_heuristic))
    print("Rule 2 Average Utility w Travel Consideration: " +
          str(rule_2_average_with_heuristic))


def compare_rules(landscape_dim):
    """
    Compare Rule 1 and Rule 2 given a fixed landscape.
    """

    # Create landscape
    landscape = Landscape(landscape_dim)

    rule_1_observations = []
    rule_2_observations = []

    # Compute observations for rule 1
    for i in tqdm(range(30)):

        # Find target and reset target
        rule_1_observations.append(search(landscape, 1)['t'])
        landscape.pick_new_target()

    # Compute observations for rule 2
    for i in tqdm(range(30)):

        # Find target and reset target
        rule_2_observations.append(search(landscape, 2)['t'])
        landscape.pick_new_target()

    rule_1_average = sum(rule_1_observations)/len(rule_1_observations)
    rule_2_average = sum(rule_2_observations)/len(rule_2_observations)

    print("Rule 1 Average Observations: "+str(rule_1_average))
    print("Rule 2 Average Observations: "+str(rule_2_average))


def search(landscape, rule, consider_travel=False):
    """
    Search for the target in a landscape. The search will require a rule to decide which cells to search.
    The heuristic will be used in addition to the rule if chosen.
    """

    # Create belief state and set initial belief probabilities
    belief_state = BeliefGrid(landscape)

    prob_of_finding_grid_states = []
    prob_target_present_grid_states = []

    # Keep running the process until target if found
    while True:

        # Increase number of actions and observation time
        belief_state.num_actions += 1
        belief_state.t += 1

        # Choose query location based on rule
        query_loc = belief_state.choose_query_loc(rule, consider_travel)

        # Update current location
        belief_state.current_loc = query_loc

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

            # Append new belief states to output : THIS IS USED FOR PLOTTING. NOT CRUCIAL TO ALGORITHM
            prob_of_finding_grid_states.append(
                belief_state.get_prob_finding_grid())

            prob_target_present_grid_states.append(
                belief_state.get_prob_present_grid())

    return {"t": belief_state.t, "num_actions": belief_state.num_actions, "prob_of_finding_grid_states": prob_of_finding_grid_states, "prob_target_present_grid_states": prob_target_present_grid_states}


def draw_heatmap(belief_states, dim):

    fig = plt.figure()

    vmin = 0
    vmax = max([state.max() for state in belief_states])

    # fig, ax = plt.subplots(1, 2)
    # sns.countplot(df['batting'], ax=ax[0])
    # sns.countplot(df['bowling'], ax=ax[1])
    # fig.show()

    def init():
        plt.clf()
        sns.heatmap(np.zeros((dim, dim)), square=True)

    def animate(state):
        plt.clf()
        sns.heatmap(state,  square=True, vmax=vmax, cmap="PiYG")

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=belief_states, repeat=True, interval=1)

    plt.show()


if __name__ == "__main__":
    main()
