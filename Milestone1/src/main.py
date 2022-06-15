import pandas as pd
import numpy as np
import math
import time
import os
from functools import reduce
import re
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="cmpt459")

#1.2 Clean age column

def clean_data(df):

  df.age.astype(str)

  # get all non empty age rows
  non_age_empty_cases = df.dropna(subset=['age'])

  # clean up age ranges
  mask = (non_age_empty_cases['age'].str.contains('-'))

  # convert age range to the middle age value
  def range_to_age(row):
    age_range = row.age.replace(' ', '').split('-')
    if any((not age.isnumeric() or age == '') for age in age_range):
     for age in age_range:
       if age.isnumeric():
         return age
    else:
      mean_age = reduce(lambda a, b: int(a) + int(b), age_range) / len(age_range)
      return mean_age 

  non_age_empty_cases.loc[mask, 'age'] = non_age_empty_cases[mask].apply(lambda row: range_to_age(row), axis=1)

  # convert babies/month old children to 1 years old if months < 12 
  non_age_empty_cases['age'] = non_age_empty_cases.age.astype(str)
  month_mask = (non_age_empty_cases['age'].str.contains('month'))

  def months_to_years(row):
    months = int(re.sub('[^0-9]','', row.age))
    return 1 if months < 12 else months/12

  non_age_empty_cases.loc[month_mask, 'age'] = non_age_empty_cases[month_mask].apply(lambda row: months_to_years(row), axis=1)

  # remove other symbols
  non_age_empty_cases['age'] = non_age_empty_cases.age.str.replace('+', '', regex=False)

  non_age_empty_cases['age'] = non_age_empty_cases.age.astype(float)

  # get all missing age values and fill them with mean age of the rest of the rows
  mean_age = non_age_empty_cases['age'].mean()

  empty_age_cases = df[df.age.isna()]
  empty_age_cases['age'] = mean_age
  empty_age_cases['age'] = empty_age_cases.age.astype(float)

  df = pd.concat([non_age_empty_cases, empty_age_cases], ignore_index=True)

  # impute missing sex

  non_empty_sex =  df.dropna(subset=['sex'])
  empty_sex = df[df.sex.isna()]

  # shuffle rows
  empty_sex = empty_sex.sample(frac=1)

  # assign half of empty sex cells as male, other half as female
  half_row_count = int(len(empty_sex.index)/2)

  empty_sex.iloc[:half_row_count].sex = 'female'
  empty_sex.iloc[half_row_count:].sex = 'male'

  #merge dfs again
  df = pd.concat([non_empty_sex, empty_sex])

  # shuffle rows
  df = df.sample(frac=1)

  # impute missing country

  # drop where countries and lat long are missing
  df = df.dropna(subset=['country', 'latitude', 'longitude'], how='all')
  
  non_empty_country =  df.dropna(subset=['country'])
  empty_country = df[df.country.isna()]
  
  def get_country(row):
    location = geolocator.reverse(','.join([str(row.latitude), str(row.longitude)]), language='en')
    return location.address.split(",")[-1]
  
  empty_country['country'] = empty_country.apply(func=get_country, axis=1)

  #merge
  df = pd.concat([non_empty_country, empty_country])

  return df

#1.4: 

def transform_counties(df):
  us_mask = (df['Country_Region'] == 'US')
  non_us_country = df[~us_mask]
 
  us_country = df[us_mask]

  us_country = us_country.groupby(['Province_State']).mean().reset_index()

  us_country['Country_Region'] = 'US'
  us_country['Combined_Key'] = us_country['Province_State'].apply(lambda province : province + ", US")
  return pd.concat([us_country, non_us_country])

#1.5: merge location and cases_train

def merge_df_location(cases_df, location_df):
  province_location_df = location_df.dropna(subset=['Province_State'])
  empty_province_location_df = location_df[location_df.Province_State.isna()]
  merged_dataset_province = pd.merge(cases_df, province_location_df, left_on=['province'], right_on=['Province_State'], how="inner")
  merged_dataset_empty_province = pd.merge(cases_df, empty_province_location_df, left_on=['country'], right_on=['Country_Region'], how="inner")
  merged_dataset = pd.concat([merged_dataset_province, merged_dataset_empty_province], ignore_index=True)
  # drop useless columns
  merged_dataset.drop(['Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Last_Update'], axis=1)
  return merged_dataset

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


def main():
  cases_df = pd.read_csv('../data/cases_train.csv')

  cases_df.drop('additional_information',axis = 1,inplace = True)
  cases_df.drop('source',axis = 1,inplace = True)

  test_df = pd.read_csv('../data/cases_test.csv')

  test_df.drop('additional_information',axis = 1,inplace = True)
  test_df.drop('source',axis = 1,inplace = True)

  location_df = pd.read_csv('../data/location.csv')

  clean_cases_df = clean_data(cases_df)

  clean_test_df = clean_data(test_df)

  location_df = transform_counties(location_df)

  merged_dataset_train = merge_df_location(clean_cases_df, location_df)
  
  merged_dataset_train = process_outlier(merged_dataset_train, location_df)

  merged_dataset_train.to_csv("../results/cases_train_processed.csv")
  
  merged_test = merge_df_location(clean_test_df, location_df)

  merged_test.to_csv("../results/cases_test_processed.csv")

  location_df.to_csv("../results/location_transformed.csv")

if __name__ == "__main__":
  main()