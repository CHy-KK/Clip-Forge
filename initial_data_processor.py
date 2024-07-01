import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA 
import logging
import torch
import ast
import pandas as pd
from sklearn.manifold import TSNE  
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# create csv
data_embedding = []
# processed_filepath = './processed_voxel_image'
processed_filepath = './processed_voxel_image_clip'
# processed_filepath = './processed_voxel_image_simple'

tsne_voxel = TSNE(n_components=2, random_state=42)
tsne_clip = TSNE(n_components=2, random_state=42)
voxel_embs = np.empty(shape=[0,128],dtype=float)
clip_embs = np.empty(shape=[0,512],dtype=float)
label = []
with open (processed_filepath + '/initial_text_query.json', 'r') as fj:
  init_dict = json.load(fj)
  for key_text, val_emb_list in init_dict.items():
    print(key_text)
    print(len(val_emb_list))
    for i in range(len(val_emb_list)):
      # 类别，命名，voxel embedding，clip image embedding
      data_embedding.append([key_text, key_text + ' ' + str(i), val_emb_list[i][0], val_emb_list[i][1]])
      print (len(val_emb_list[i][0]))
      print (len(val_emb_list[i][1]))
      voxel_embs = np.append(voxel_embs, np.array(val_emb_list[i][0], ndmin=2).astype(np.float), axis=0)
      clip_embs = np.append(clip_embs, np.array(val_emb_list[i][1], ndmin=2).astype(np.float), axis=0)
      label.append(key_text)
    #   with open(processed_filepath + '/' + key_text + ' ' + str(i), 'w', encoding='utf-8') as processed_data:
    #     # 体素，图片
    #     processed_data.write(json.dumps({'voxel': val_emb_list[i][2], 'image': val_emb_list[i][3]}))

print (clip_embs)
shape_embs_position = tsne_voxel.fit_transform(voxel_embs)
clip_embs_position = tsne_clip.fit_transform(clip_embs)
print(voxel_embs.shape)
print(clip_embs.shape)

for i in range(len(data_embedding)):
    # if (shape_embs_position[i].tolist() != clip_embs_position[i].tolist()):
        # print ('diff')
    data_embedding[i].append(shape_embs_position[i].tolist())
    data_embedding[i].append(clip_embs_position[i].tolist())

simMat = linkage(voxel_embs, method='ward', metric='euclidean')
fig = plt.figure(figsize=(8,5))
dn = dendrogram(simMat, labels=label)
plt.savefig(processed_filepath + '/dendrogram_graph')
print (dn)

with open(processed_filepath + '/init_data_voxel_image_tsne.csv', 'w', newline='') as fc:
  writer = csv.writer(fc)
  writer.writerows(data_embedding)



{
    'icoord': [[5.0, 5.0, 15.0, 15.0], [25.0, 25.0, 35.0, 35.0], [10.0, 10.0, 30.0, 30.0], [45.0, 45.0, 55.0, 55.0], 
            [65.0, 65.0, 75.0, 75.0], [50.0, 50.0, 70.0, 70.0], [105.0, 105.0, 115.0, 115.0], [95.0, 95.0, 110.0, 110.0], 
            [85.0, 85.0, 102.5, 102.5], [60.0, 60.0, 93.75, 93.75], [20.0, 20.0, 76.875, 76.875]], 
    'dcoord': [[0.0, 0.0, 0.0, 0.0], [0.0, 15.678327139954764, 15.678327139954764, 0.0], [0.0, 27.256331439449397, 27.256331439449397, 15.678327139954764], 
            [0.0, 0.0, 0.0, 0.0], [0.0, 9.624373862842635, 9.624373862842635, 0.0], [0.0, 15.079784119478944, 15.079784119478944, 9.624373862842635], 
            [0.0, 0.0, 0.0, 0.0], [0.0, 14.83561757278, 14.83561757278, 0.0], [0.0, 15.844805348733685, 15.844805348733685, 14.83561757278], 
            [15.079784119478944, 46.550743072231654, 46.550743072231654, 15.844805348733685], [27.256331439449397, 47.31132268844892, 47.31132268844892, 46.550743072231654]], 
    'ivl': ['cabinet', 'cabinet', 'cabinet', 'cabinet', 'bench', 'bench', 'bench', 'bench', 'plane', 'plane', 'plane', 'plane'], 
    'leaves': [8, 11, 9, 10, 5, 7, 4, 6, 0, 1, 2, 3], 
    # 'color_list': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3', 'C0', 'C0'], 
    # 'leaves_color_list': ['C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3', 'C3']
}



# with open(processed_filepath + '/' + 'bench 0', 'r', encoding='utf-8') as processed_data:
#     # 体素，图片
#     data = json.load(processed_data)
#     print(type(data['voxel']))
#     print(type(data['image']))
