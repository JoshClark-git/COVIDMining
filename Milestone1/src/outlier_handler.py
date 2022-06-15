import pandas as pd


# simply use this function on merged train_dataset as input_df,
# as well as dataframe from location.csv
def process_outlier(input_df, location_df):
    #1 remove date confirmed after June 2020
    input_df['date_confirmation'] = pd.to_datetime(
        input_df['date_confirmation'], format='%d.%m.%Y', errors='coerce')
    df = input_df[input_df['date_confirmation'] < '2020-06-01']
    #2 remove rows with abnormal Case-Fatality ratio
    lower, upper = get_case_fatality_range(location_df)
    df = df[(df['Case-Fatality_Ratio'] > lower) &
            (df['Case-Fatality_Ratio'] < upper)]

    return df

# function to calculate filter range
def get_case_fatality_range(location_df):
    mean_val = location_df['Case-Fatality_Ratio'].mean()
    sd = location_df['Case-Fatality_Ratio'].std()
    lower_bound = mean_val-3*sd
    upper_bound = mean_val+3*sd
    if (lower_bound < 0):
        lower_bound = 0
    return (lower_bound, upper_bound)
