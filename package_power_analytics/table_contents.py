from docxtpl import DocxTemplate, InlineImage
import pandas as pd


def table_contents(filename):
    table_contents = []
    data = pd.read_excel(filename)
    for key, row in data.iterrows():
        table_contents.append({'Date': str(row[0]),
                               'Value_act_power': round(row[1], 3),
                               'Value_react_power': round(row[2], 3),
                               })
        if key == 35:
            break
    return table_contents

def table_dynamic(data):
    key_dict = ['v'+ str(i) for i in range(len(data.columns))]
    table_contents = []
    for key, row in data.iterrows():
        dict = {}
        for i, key in enumerate(key_dict):
            dict[key] = str(row[i])
        table_contents.append(dict)
    return table_contents
