
import pickle 
import pandas as pd
import os
DATA_DIR = "data"


def split_to_types():
    dic = {}

    c = pd.read_pickle("data/samples.pkl")
    for index, row in c.iterrows():
        typee = "human_yac" if (row['chr'][:2]!="JS" and row['chr'][:2]!="NC") else "others"
        if typee not in dic:
            dic.update({typee: c.loc[[index]]})
        else:
            temp = pd.concat([dic.get(typee), c.loc[[index]]], ignore_index=True)
            dic.update({typee: temp})
    file_path = f"{DATA_DIR}/type_splits"
    os.mkdir(file_path)


    with open(f"{DATA_DIR}/type_splits.txt",'a') as file:
        for key,value in dic.items():
            file.write(f"\"{key}\"\n")

    for key,value in dic.items():
        with open(f"{file_path}/{key}.pkl", 'wb') as file:
            pickle.dump(value, file)




def split_to_categories():
    dic = {}

    c = pd.read_pickle("data/samples.pkl")
    for index, row in c.iterrows():
        if row['chr'] not in dic:
            dic.update({row['chr']: c.loc[[index]]})
        else:
            temp = pd.concat([dic.get(row['chr']), c.loc[[index]]], ignore_index=True)
            dic.update({row['chr']: temp})
    file_path = f"{DATA_DIR}/categories"
    os.mkdir(file_path)


    with open(f"{DATA_DIR}/categories.txt",'a') as file:
        for key,value in dic.items():
            file.write(f"\"{key}\"\n")

    for key,value in dic.items():
        with open(f"{file_path}/{key}.pkl", 'wb') as file:
            pickle.dump(value, file)



split_to_categories()
split_to_types()