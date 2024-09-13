import os
import pandas as pd

def read_and_structure_csv_files(folder_path):
    # Initialize an empty list to store DataFrames
    data_frames = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV
        # if file_name == 'Base Model.csv' or 'Default' in file_name or 'Conn' in file_name:
        if file_name == 'Base Model.csv' or 'Conn' in file_name:
            continue
        if file_name.endswith('.csv'):
            print(file_name)
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df.rename(columns={'True': 'Tricked'}, inplace=True)
            df['Identifier'] = df['Identifier'].replace('Unknown', 'BPDA')
            # Append the DataFrame to the list
            data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    #combined_df = pd.concat(data_frames, ignore_index=True)

    return data_frames

# Specify the folder path containing the CSV files
folder_path = 'csv_res_ultraHard/'

# Call the function to read and structure CSV files
structured_data = read_and_structure_csv_files(folder_path)

base_model_df = pd.read_csv(f'{folder_path}Base Model.csv')
base_model_df.rename(columns={'True': 'Tricked'}, inplace=True)
base_model_df['Identifier'] = base_model_df['Identifier'].replace('Unknown', 'BPDA')
# print(base_model_df.dtypes)
# print(base_model_df)

# Filter the Base Model DataFrame
base_model_filtered = base_model_df[
    (base_model_df.iloc[:, 0] == 'Unknown') &
    (base_model_df.iloc[:, 2] == True) &
    (base_model_df.iloc[:, 3] > 0)
    ]

print(base_model_filtered)

# Filter the combined DataFrame
# combined_other_filtered = structured_data[
#     (structured_data.iloc[:, 2] == 'False') &
#     (structured_data.iloc[:, 3] > 0)
#     ]


combined_other_filtered = [data[
                               (data.iloc[:, 0] == 'Unknown') &
                               (data.iloc[:, 2] == False) &
                               (data.iloc[:, 3] > 0)
                               ] for data in structured_data]

print(combined_other_filtered)



# Find rows in base_model_filtered where "Evaluation" and "Identifier" appear in all other DataFrames
# Find rows in base_model_filtered with the most matches in other DataFrames
def count_matches(row):
    return sum(
        any(
            (other_df.iloc[:, 0] == row.iloc[0]) &
            (other_df.iloc[:, 1] == row.iloc[1])
        ) for other_df in combined_other_filtered
    )

base_model_filtered['match_count'] = base_model_filtered.apply(count_matches, axis=1)
max_matches = base_model_filtered['match_count'].max()
most_matched_rows = base_model_filtered[base_model_filtered['match_count'] == max_matches]


print(most_matched_rows)




# # Find common "Evaluation" numbers and identifiers in both filtered DataFrames
# common_evaluations = base_model_filtered.merge(
#     combined_other_filtered,
#     on=['Evaluation', 'Identifier'],  # Assuming 'Evaluation' and 'Identifier' are the column names
#     suffixes=('_base', '_other')
# )
#
# # Extract relevant columns to display
# result_df = common_evaluations[['Evaluation', 'Identifier']]

# print(common_evaluations)


#%%
