import pickle 
import pandas as pd
c = pd.read_pickle("categorized/759h10_NC_000004.12.pkl")
print(c.columns)
print(c.iloc[:])
