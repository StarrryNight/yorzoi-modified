import pickle 
import pandas as pd
c = pd.read_pickle("samples.pkl")
print(c.columns)
print(c.iloc[0])
