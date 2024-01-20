import pandas as pd

# Sample DataFrame with missing values and duplicates
data = {
    'Feature1': [1, 2, None, 4, 5, 2],
    'Feature2': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Target': [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Drop duplicates
df.drop_duplicates(inplace=True)
print("\nDataFrame after dropping duplicates:")
print(df)

# Fill missing values (replace with mean for numerical features and mode for categorical features)
df['Feature1'].fillna(df['Feature1'].mean(), inplace=True)
print("\nDataFrame after filling missing values:")
print(df)

# Sample DataFrame with non-UTF-8 symbols and zeros
data = {
    'Text': ['Hello', 'World', 'Привет', '123', 'Non-UTF-8 Symbol: \x80', 'Zero Row', 'Another Text'],
    'Value': [1, 2, 0, 4, 0, 0, 7]
}

df = pd.DataFrame(data)

# Remove non-UTF-8 symbols from the 'Text' column
df['Text'] = df['Text'].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))

# Remove rows with zeros in the 'Value' column
df = df[df['Value'] != 0]
print("\nDataFrame after removing non-UTF-8 symbols and zeros:")
print(df)

# Save the cleaned DataFrame to a file
cleaned_filename = 'cleaned_data.csv'
df.to_csv(cleaned_filename, index=False)

# Display the cleaned DataFrame
print("\nCleaned DataFrame:")
print(df)
