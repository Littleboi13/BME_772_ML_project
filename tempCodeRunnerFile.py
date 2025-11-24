# Print the entire CSV (ensure pandas will not truncate the output)
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df.to_string(index=False))
