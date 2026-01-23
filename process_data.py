
import pickle 
import pandas as pd


dic = {}
c = pd.read_pickle("samples.pkl")
for index, row in c.iterrows():
    if row['chr'] not in dic:
        dic.update({row['chr']: c.loc[[index]]})
    else:
        temp = pd.concat([dic.get(row['chr']), c.loc[[index]]], ignore_index=True)
        dic.update({row['chr']: temp})
file_path = "categorized"

with open(f"category_names.txt",'a') as file:
    for key,value in dic.items():
        file.write(f"\"{key}\"\n")

'''
for key,value in dic.items():
    with open(f"{file_path}/{key}.pkl", 'wb') as file:
        pickle.dump(value, file)
'''


