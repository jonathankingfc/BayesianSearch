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

    compare_rules_with_moving_target(9)


def compare_heatmaps(landscape_dim):

    landscape = Landscape(landscape_dim)

    belief_states = search(landscape, 1, consider_travel=False)[
        'prob_of_target_present_grid_states']

    draw_heatmap(belief_states, landscape_dim)


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

    print("Rule 1 Average Utility \n w/o Travel Consideration: " +
          str(rule_1_average_without_heuristic))
    print("Rule 1 Average Utility \n w Travel Consideration: " +
          str(rule_1_average_with_heuristic))

    print("Rule 2 Average Utility \n w/o Travel Consideration: " +
          str(rule_2_average_without_heuristic))
    print("Rule 2 Average Utility \n w Travel Consideration: " +
          str(rule_2_average_with_heuristic))

    x = np.arange(4)
    averages = [rule_1_average_with_heuristic, rule_1_average_without_heuristic,
                rule_2_average_with_heuristic, rule_2_average_without_heuristic]

    fig, ax = plt.subplots()

    plt.bar(x, averages)
    plt.xticks(x, ('Rule 1 Average Utility \nw Travel Consideration', 'Rule 1 Average Utility \nw/o Travel Consideration',
                   'Rule 2 Average Utility \nw Travel Consideration', 'Rule 2 Average Utility \nw/o Travel Consideration'))

    plt.ylabel("Actions Needed")
    plt.title(
        "Average Number of Actions Needed \nto Find Target with and without Utility Heuristic")
    plt.show()


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

    x = np.arange(2)
    averages = [rule_1_average, rule_2_average]

    fig, ax = plt.subplots()

    plt.bar(x, averages)
    plt.xticks(x, ('Rule 1', 'Rule 2'))
    plt.ylabel("Observations")
    plt.title("Average Number of Observations Needed to Find Target")
    plt.show()


def compare_rules_with_moving_target(landscape_dim):
    """
    Compare rules with moving target.
    """

    # Create landscape
    landscape = Landscape(landscape_dim)

    # Create space for utility when using tracker vs not using it
    rule_1_num_actions_without_tracker = []
    rule_1_num_actions_with_tracker = []

    # Create space for utility when using tracker vs not using it
    rule_2_num_actions_without_tracker = []
    rule_2_num_actions_with_tracker = []

    # Compute observations for rule 1
    for i in tqdm(range(50)):

        # Find target without using tracker
        result = search_moving(
            landscape, 1, consider_travel=True, use_tracker=False)
        rule_1_num_actions_without_tracker.append(result['num_actions'])

        # Find target with tracker
        result = search_moving(
            landscape, 1, consider_travel=True, use_tracker=True)
        rule_1_num_actions_with_tracker.append(result['num_actions'])

        # Reset target location
        landscape.pick_new_target()

    # Compute observations for rule 2
    for i in tqdm(range(50)):

        # Find target without using tracker
        result = search_moving(
            landscape, 2, consider_travel=True, use_tracker=False)
        rule_2_num_actions_without_tracker.append(result['num_actions'])

        # Find target with tracker
        result = search_moving(
            landscape, 2, consider_travel=True, use_tracker=True)
        rule_2_num_actions_with_tracker.append(result['num_actions'])

        # Reset target location
        landscape.pick_new_target()

    # Get average observations for each rule with and without using the heuristic
    rule_1_average_without_tracker = sum(
        rule_1_num_actions_without_tracker)/len(rule_1_num_actions_without_tracker)
    rule_2_average_without_tracker = sum(
        rule_2_num_actions_without_tracker)/len(rule_2_num_actions_without_tracker)

    # Get average observations for each rule with and without using the heuristic
    rule_1_average_with_tracker = sum(
        rule_1_num_actions_with_tracker)/len(rule_1_num_actions_with_tracker)
    rule_2_average_with_tracker = sum(
        rule_2_num_actions_with_tracker)/len(rule_2_num_actions_with_tracker)

    x = np.arange(4)
    averages = [rule_1_average_with_tracker, rule_1_average_without_tracker,
                rule_2_average_with_tracker, rule_2_average_without_tracker]

    fig, ax = plt.subplots()

    plt.bar(x, averages)
    plt.xticks(x, ('Rule 1 Average Utility \nw Tracker', 'Rule 1 Average Utility \nw/o Tracker',
                   'Rule 2 Average Utility \nw Tracker', 'Rule 2 Average Utility \nw/o Tracker'))

    plt.ylabel("Actions Needed")
    plt.title(
        "Average Number of Actions Needed \nto Find Target with and without Tracker")
    plt.show()


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


def search_moving(landscape, rule, consider_travel=False, use_tracker=False):
    """
    Search for the target in a landscape. The search will require a rule to decide which cells to search.
    The heuristic will be used in addition to the rule if chosen. In this function, the target will be moving.
    After every
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

        # Our tracker doesn't say anything until the target moves for the first time
        target_not_in_terrain = None

        # Choose query location based on rule
        if use_tracker:
            query_loc = belief_state.choose_query_loc(
                rule, consider_travel, target_not_in_terrain)
        else:
            query_loc = belief_state.choose_query_loc(
                rule, consider_travel)

        # Update current location
        belief_state.current_loc = query_loc

        # Resolve query
        result = landscape.resolve_query(query_loc)

        # If target was found
        if result == True:
            break

        # Otherwise update belief grid
        else:

            # Move the target and get location that it is not present
            target_not_in_terrain = landscape.move_target_to_neighbor()

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
