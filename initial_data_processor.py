import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA 
import logging
import torch
import ast
import pandas as pd

# create csv
data_embedding = []
# processed_filepath = './processed_voxel_image'
processed_filepath = './processed_voxel_image_clip'
# processed_filepath = './processed_voxel_image_simple'
with open (processed_filepath + '/initial_text_query.json', 'r') as fj:
  init_dict = json.load(fj)
  for key_text, val_emb_list in init_dict.items():
    print(key_text)
    print(len(val_emb_list))
    for i in range(len(val_emb_list)):
      # 类别，命名，embedding，clip image embedding
      data_embedding.append([key_text, key_text + ' ' + str(i), val_emb_list[i][0], val_emb_list[i][1]])
      # print(len(val_emb_list[i]))
      with open(processed_filepath + '/' + key_text + ' ' + str(i), 'w', encoding='utf-8') as processed_data:
        # 体素，图片
        processed_data.write(json.dumps({'voxel': val_emb_list[i][2], 'image': val_emb_list[i][3]}))

with open(processed_filepath + '/init_data_voxel_image.csv', 'w', newline='') as fc:
  writer = csv.writer(fc)
  writer.writerows(data_embedding)






# with open(processed_filepath + '/' + 'bench 0', 'r', encoding='utf-8') as processed_data:
#     # 体素，图片
#     data = json.load(processed_data)
#     print(type(data['voxel']))
#     print(type(data['image']))
