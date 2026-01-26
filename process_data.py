
import pickle 
import pandas as pd
import os

dic = {}
c = pd.read_pickle("samples.pkl")
for index, row in c.iterrows():
    typee = "human_yac" if (row['chr'][:2]!="JS" and row['chr'][:2]!="NC") else "others"
    if typee not in dic:
        dic.update({typee: c.loc[[index]]})
    else:
        temp = pd.concat([dic.get(typee), c.loc[[index]]], ignore_index=True)
        dic.update({typee: temp})
file_path = "type_splits"
os.mkdir(file_path)


with open(f"type_names.txt",'a') as file:
    for key,value in dic.items():
        file.write(f"\"{key}\"\n")

for key,value in dic.items():
    with open(f"{file_path}/{key}.pkl", 'wb') as file:
        pickle.dump(value, file)



