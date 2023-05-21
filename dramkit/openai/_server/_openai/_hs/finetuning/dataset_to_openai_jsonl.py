# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
from dramkit.gentools import sort_dict
from dramkit.iotools import read_lines, write_txt


def _trans(x):
    res = eval(x)
    res = sort_dict(res, by='key', reverse=True)
    res = str(res).replace("'", '"')
    return res
    

if __name__ == '__main__':
    lines = read_lines('./openai_finetuning_dataset.txt')
    
    data = lines.copy()
    for k in tqdm(range(len(data))):
        data[k] = _trans(data[k])
    write_txt(data, './openai_finetuning_dataset.jsonl',
              encoding='utf-8')
    
    df = [eval(x) for x in data]
    df = pd.DataFrame(df)
    df.to_csv('openai_finetuning_dataset.csv', index=None,
              encoding='utf-8')