import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from itertools import chain

st.set_page_config(layout="wide")

DATA_DIR = 'data'

TEAMS_OF_INTEREST = ["Finland", "Guernsey", "Bulgaria", "Estonia", "Romania", "Malta"]


@st.cache(allow_output_mutation=True)
def load_filtered_matches(data_dir, teams):
    filtered_matches = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as file:
                match_data = json.load(file)
                if any(team in match_data['info']['teams'] for team in teams):
                    filtered_matches.append(file_name)
    return filtered_matches

filtered_matches = load_filtered_matches(DATA_DIR, TEAMS_OF_INTEREST)


# Function to get a summary of the match
def get_match_summary(match_data):
    return {
        'Date': match_data['info']['dates'][0],
        'Venue': match_data['info']['venue'],
        'City': match_data['info'].get('city', 'N/A'),
        'Teams': ' vs '.join(match_data['info']['teams']),
        'Toss': match_data['info']['toss']['winner'] + ' won the toss',
        'Winner': match_data['info']['outcome'].get('winner', 'No result'),
        'Match Number': match_data['info'].get('match_type_number', 'N/A')
    }


def display_match_summary(summary):
    # Convert the summary dictionary to a DataFrame for a single row
    summary_df = pd.DataFrame([summary])
    # Display the DataFrame as a table in Streamlit
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


def get_innings_summary(data):
    innings_summaries = []
    for innings in data['innings']:
        team = innings['team']
        total_runs = 0
        total_overs = len(innings['overs'])
        phase1_runs = 0  # Overs 0-6
        phase2_runs = 0  # Overs 15-20
        runs_per_over = []
        cumulative_runs = []
        cumulative_wickets = 0
        wickets_per_over = []

        for over in innings['overs']:
            over_runs = 0
            over_wickets = 0
            for delivery in over['deliveries']:
                over_runs += delivery['runs']['total']
                if 'wickets' in delivery:  # Checking if a wicket has been taken in this delivery
                    over_wickets += len(delivery['wickets'])
            total_runs += over_runs
            cumulative_wickets += over_wickets
            runs_per_over.append(over_runs)
            wickets_per_over.append(over_wickets)
            cumulative_runs.append(total_runs if not cumulative_runs else cumulative_runs[-1] + over_runs)

            over_number = over['over']
            if over_number < 6:
                phase1_runs += over_runs
            elif 15 <= over_number < 20:
                phase2_runs += over_runs

        innings_summaries.append({
            "Team": team,
            "Total overs played": total_overs,
            "Total runs scored": total_runs,
            "Powerplay runs scored": phase1_runs,
            "Backend runs scored": phase2_runs,
            "Total wickets taken": cumulative_wickets,
            "cumulative_runs": cumulative_runs,
            "runs_per_over": runs_per_over,
            "cumulative_wickets": cumulative_wickets,
            "wickets_per_over": wickets_per_over
        })

    return innings_summaries


def plot_innings_summaries(innings_summary):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Define the range for x-ticks
    x_ticks = np.arange(0, 21, 2)  # From 0 to 20 inclusive, step of 2

    for i, summary in enumerate(innings_summary):
        # Cumulative Runs Plot
        axs[0].plot(summary['cumulative_runs'], label=summary['Team'], color=colors[i])
        axs[0].set_xticks(x_ticks)  # Set the x-ticks
        axs[0].set_xlim([0, 20])  # Set the x-axis limit
        axs[0].set_xlabel('Overs')
        axs[0].set_ylabel('Cumulative Runs')
        axs[0].set_title('Overs vs Cumulative Runs Scored')
        axs[0].legend()

        # Runs per Over Plot - Clustered Bars
        bar_width = 0.35
        index = np.arange(len(summary['runs_per_over']))
        axs[1].bar(index + bar_width*i, summary['runs_per_over'], bar_width, label=summary['Team'], alpha=0.5, color=colors[i])
        axs[1].set_xticks(x_ticks)  # Set the x-ticks
        axs[1].set_xlim([0, 20])  # Set the x-axis limit
        axs[1].set_xlabel('Overs')
        axs[1].set_ylabel('Runs per Over')
        axs[1].set_title('Overs vs Runs per Over')
        axs[1].legend()

    st.pyplot(fig)


@st.cache(allow_output_mutation=True)
def calculate_aggregates(data_dir, teams_of_interest):
    # Initialize the dictionary to hold aggregate data
    aggregates = {team: {'matches': 0, 'pp_runs': 0, 'be_runs': 0, 'pp_wickets': 0, 'be_wickets': 0} for team in teams_of_interest}

    # Process each file in the directory
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for innings in data.get('innings', []):
                team = innings['team']
                if team in teams_of_interest:
                    # Update the match count for the team
                    aggregates[team]['matches'] += 1

                    # Initialize the count of runs and wickets for this innings
                    pp_runs, be_runs, pp_wickets, be_wickets = 0, 0, 0, 0

                    # Go through each over in the innings
                    for over in innings['overs']:
                        over_index = over['over']
                        for delivery in over['deliveries']:
                            # Count runs in powerplay (overs 0-6)
                            if over_index < 6:
                                pp_runs += delivery['runs']['total']
                            # Count runs in the backend (overs 15-20)
                            elif 15 <= over_index < 20:
                                be_runs += delivery['runs']['total']

                            # Count wickets
                            if 'wickets' in delivery:
                                wicket_count = len(delivery['wickets'])
                                if over_index < 6:
                                    pp_wickets += wicket_count
                                elif 15 <= over_index < 20:
                                    be_wickets += wicket_count

                    # Aggregate the counts
                    aggregates[team]['pp_runs'] += pp_runs
                    aggregates[team]['be_runs'] += be_runs
                    aggregates[team]['pp_wickets'] += pp_wickets
                    aggregates[team]['be_wickets'] += be_wickets

    # Calculate averages
    for team_stats in aggregates.values():
        if team_stats['matches'] > 0:
            team_stats['Avg. powerplay runs scored'] = round(team_stats['pp_runs'] / team_stats['matches'], 1)
            team_stats['Avg. backend runs scored'] = round(team_stats['be_runs'] / team_stats['matches'], 1)
            team_stats['Avg. powerplay wickets taken'] = round(team_stats['pp_wickets'] / team_stats['matches'], 1)
            team_stats['Avg. backend wickets taken'] = round(team_stats['be_wickets'] / team_stats['matches'], 1)

    return aggregates


# Get the aggregated data
aggregated_data = calculate_aggregates(DATA_DIR, TEAMS_OF_INTEREST)

# Convert the aggregated data to a DataFrame for display
aggregated_df = pd.DataFrame.from_dict(aggregated_data, orient='index',
                                       columns=['matches', 'Avg. powerplay runs scored', 'Avg. backend runs scored', 'Avg. powerplay wickets taken', 'Avg. backend wickets taken'])


# Streamlit UI
st.title("Summarising matches of teams of interest")
st.sidebar.header("Select a match to break down:")

# Dropdown to select a match
selected_file = st.sidebar.selectbox("", filtered_matches)

# Display the match data
if selected_file:
    match_file = os.path.join(DATA_DIR, selected_file)
    with open(match_file, 'r') as file:
        match_data = json.load(file)
        match_summary = get_match_summary(match_data)

        innings_summary = get_innings_summary(match_data)

        st.write("Aggregate stats for teams of interest:")
        #st.table(aggregated_df.reset_index().rename(columns={'index': 'Team'}))
        st.dataframe(aggregated_df.reset_index().rename(columns={'index': 'Team'}), hide_index=True, use_container_width=True)

        st.markdown("""
        <br><br><br> 
        """, unsafe_allow_html=True)

        st.write("Selected match:")
        display_match_summary(match_summary)
        #st.table(match_summary)

        st.write("Innings breakdown for selected match:")
        innings_summary_df = pd.DataFrame(innings_summary).drop(columns=['cumulative_runs', 'runs_per_over', 'cumulative_wickets', 'wickets_per_over'], errors='ignore')
        st.dataframe(innings_summary_df, hide_index=True, use_container_width=True)

        plot_innings_summaries(innings_summary)

# Custom CSS for improved font in tables
st.markdown("""
<style>
table {
    font-family: Arial, sans-serif;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)
