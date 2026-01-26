import pickle 
import pandas as pd
c = pd.read_pickle("type_splits/others.pkl")
print(c.columns)
print(c)
