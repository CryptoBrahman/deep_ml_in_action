import pandas as pd

data = {
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value1': [10, 20, 15, 25, 12, 18],
    'Value2': [5, 8, 7, 9, 4, 6]
}

df = pd.DataFrame(data)

agg_df = df.groupby('Category').agg({
    'Value1': ['sum', 'mean'],
    'Value2': 'max',
    'Value3': lambda x: x.mean()
}).reset_index()

print("Aggregated DataFrame:")
print(agg_df)
