import pandas as pd

def convert_to_numeric(value):
    if isinstance(value, str) and value == "Final FAR ERROR":
        return 0
    try:
        numeric_value = float(value)
        if 0 <= numeric_value <= 1:
            return numeric_value
        else:
            return 0  # For values outside the range
    except ValueError:
        return 0  # For non-convertible values




data = pd.read_csv("df_anonymized.csv")
print(data.head())

desired_columns = [
    "uniq_patient_num", "Index_f", "Age", "isFemale",
    "prev_left_angle", "prev_left_cylinder", "prev_right_angle", "prev_right_cylinder", "prev_right_number",
    "Ref_Objective_left_angle", "Ref_Objective_left_cylinder", "Ref_Objective_left_number",
    "Ref_Objective_right_angle", "Ref_Objective_right_cylinder", "Ref_Objective_right_number",
    "is_normal_obstetrical_history", "is_medication_sensitivity", "is_medication_background",
    "is_myopia", "is_amblyopia", "eyes_are_healthy",
    "final_far_LEFT", "final_far_RIGHT", "final_far_type", "SaveDate", "prev_left_number","is_myopia"
]

# Select the desired columns
data = data[desired_columns]
print(data.head())
data.count()

data = data.drop_duplicates(subset=['uniq_patient_num', 'SaveDate'])
data.count()

for column in data.columns:
    missing_count = data[column].isna().sum()
    print(f"{column}: {missing_count}")

data.loc[:, 'SaveDate'] = data['SaveDate'].astype(str).str.strip()

columns_to_fill = ['Ref_Objective_left_angle', 'Ref_Objective_left_cylinder', 'Ref_Objective_left_number',
                   'Ref_Objective_right_angle', 'Ref_Objective_right_cylinder', 'Ref_Objective_right_number',
                   'final_far_LEFT', 'final_far_RIGHT', 'prev_left_angle' , 'prev_left_cylinder', 'prev_left_number',
                   'prev_right_angle', 'prev_right_cylinder', 'prev_right_number']

# Iterate through each column and fill missing values based on the same Index_f and last available value
for idx, row in data.iterrows():
    for column in columns_to_fill:
        if pd.isnull(row[column]):  # Check if the cell is NaN
            same_index_rows = data[data['uniq_patient_num'] == row['uniq_patient_num']]
            last_value = same_index_rows[column].iloc[-1]  # Get the last non-NaN value
            data.at[idx, column] = last_value


data.loc[:, columns_to_fill] = data.loc[:, columns_to_fill].fillna(0)


columns_to_convert = ["final_far_LEFT", "final_far_RIGHT"]  # Replace with actual column names
data[columns_to_convert] = data[columns_to_convert].apply(lambda col: col.map(convert_to_numeric))

float_columns = ["final_far_LEFT", "final_far_RIGHT"]  # Replace with actual column names
for col in float_columns:
    data[col] = data[col].apply(lambda x: f'{float(x):.2f}' if x % 1 != 0 else str(int(x)))


mapping = {'with': 1, 'without': 0}
data['final_far_type'] = data['final_far_type'].map(mapping)

nan_count = data['final_far_type'].isna().sum()
print(f"Number of NaN values in '{'final_far_type'}': {nan_count}")

data = data.dropna(subset=['final_far_type'])

data.rename(columns={'is_myopia.1': 'is_myopia'}, inplace=True)


float_columns2 = ["final_far_type", "Ref_Objective_left_angle", "final_far_LEFT", "final_far_RIGHT", "Age"]  # Replace with actual column names
for col in float_columns2:
    data[col] = data[col].apply(lambda x: f'{float(x):.2f}' if x % 1 != 0 else str(int(x)))


count = 0
for i,idx in enumerate(data['Age']):
    if idx == 'nan':
        count += 1
        data.drop(i, inplace=True)

print(f"Age: {count} NaN values")


count = 0
for col in data:
    for i,idx in enumerate(data[col]):
        if idx == ' ':
            count += 1
            data.at[i, col] = 0


print(f"Age: {count} NaN values")

cleaned_csv_path = "cleaned_data.csv"
data.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned data saved to {cleaned_csv_path}")


data = pd.read_csv("cleaned_data.csv")
data.drop(['SaveDate', 'is_myopia.1', 'eyes_are_healthy', 'is_amblyopia'], axis=1, inplace=True)

nan_mask = pd.isna(data)
any_nan = nan_mask.any().any()
data.fillna(0, inplace=True)

cleaned_csv_path = "cleaned_data.csv"
data.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned data saved to {cleaned_csv_path}")

data = pd.read_csv("cleaned_data.csv")

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)

cleaned_csv_path = "cleaned_data.csv"
data.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned data saved to {cleaned_csv_path}")