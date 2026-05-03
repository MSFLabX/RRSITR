import pandas as pd
import numpy as np

# Load the original CSV dataset
df = pd.read_csv('RSITMD-NC-00.csv')

np.random.seed(42)
# Generate modified datasets across varying perturbation ratios
for ratio in [0.2, 0.4, 0.6, 0.8]:
    df_modified = df.copy()
    n_samples = int(len(df) * ratio)
    selected_indices = np.random.choice(df.index,size=n_samples,replace=False)
    titles_to_shuffle = df_modified.loc[selected_indices, 'title'].values
    np.random.shuffle(titles_to_shuffle)
    df_modified.loc[selected_indices, 'title'] = titles_to_shuffle
    # Export the modified dataset to a new CSV file
    output_filename = f'RSITMD-NC-0{int(ratio * 10)}.csv'
    df_modified.to_csv(output_filename, index=False)