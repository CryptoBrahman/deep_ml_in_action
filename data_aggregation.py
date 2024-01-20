import pandas as pd

# Sample DataFrame with some repeating values
data = {
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value1': [10, 20, 15, 25, 12, 18],
    'Value2': [5, 8, 7, 9, 4, 6]
}

df = pd.DataFrame(data)

# Example: Aggregate data by creating new features
agg_df = df.groupby('Category').agg({
    'Value1': ['sum', 'mean'],     # Sum and mean of 'Value1'
    'Value2': 'max',                # Maximum of 'Value2'
    'Value3': lambda x: x.mean()   # Example: Custom aggregation, mean of 'Value3' (not exist in original DataFrame)
}).reset_index()

# Display the aggregated DataFrame
print("Aggregated DataFrame:")
print(agg_df)
