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

cluster_mat = linkage(voxel_embs, method='ward', metric='euclidean')
fig = plt.figure(figsize=(8,5))
dn = dendrogram(cluster_mat, labels=label)
plt.savefig(processed_filepath + '/dendrogram_graph')
print (cluster_mat)

# 储存层次聚类的结果
# 0: 簇元素1下标
# 1: 簇元素2下标，如果为-1表示叶节点
                ##########2: 两簇的距离, -1表示叶节点
# 2: 该簇包含的所有元素数量
# 3: 该簇的voxel embedding中心点，其中簇节点的voxel embedding不参与距离的计算
dandrogramRecord = []

# 计算Ward方差最小化距离的平方
def ward_dis_square(dandroList, ele1, ele2):
    # 先判断是否为叶节点
    if (not isinstance(ele1, np.ndarray) and dandroList[ele1][1] == -1):
        ele1 = dandroList[ele1][-1]
    if (not isinstance(ele2, np.ndarray) and dandroList[ele2][1] == -1):
        ele2 = dandroList[ele2][-1]

    if isinstance(ele2, np.ndarray):
        # ele1和ele2都为ndarray
        if isinstance(ele1, np.ndarray):
            return np.sum((ele1 - ele2) ** 2)
        # ele2是ndarray，ele1为簇序号
        tmp = ele1
        ele1 = ele2
        ele2 = tmp
    if isinstance(ele1, np.ndarray) and not isinstance(ele2, np.ndarray):
        # ele1为ndarray, ele2为簇序号
        T = dandroList[ele2][2] + 1
        tl = dandroList[dandroList[ele2][0]][2]
        sl = dandroList[dandroList[ele2][1]][2]
        return ((tl + 1) * ward_dis_square(dandroList, ele1, dandroList[ele2][0]) 
                + (sl + 1) * ward_dis_square(dandroList, ele1, dandroList[ele2][1])
                - 1 * ward_dis_square(dandroList, dandroList[ele2][0], dandroList[ele2][1])) / T

    # ele1和ele2都为簇序号
    T = dandroList[ele1][2] + dandroList[ele2][2] 
    tl = dandroList[dandroList[ele2][0]][2]
    sl = dandroList[dandroList[ele2][1]][2]
    vl = dandroList[ele1][2]
    return ((tl + vl) * ward_dis_square(dandroList, ele1, dandroList[ele2][0])
            + (sl + vl) * ward_dis_square(dandroList, ele1, dandroList[ele2][1])
            - vl * ward_dis_square(dandroList, dandroList[ele2][0], dandroList[ele2][1])) / T


for i in range(len(voxel_embs)):
    dandrogramRecord.append([i, -1, 1, voxel_embs[i]])

for i in range(len(cluster_mat)):
    idx0 = int(cluster_mat[i][0])
    idx1 = int(cluster_mat[i][1])
    cluster_eles = int(cluster_mat[i][3])
    cluster_centroid0 = (dandrogramRecord[idx0][3] * dandrogramRecord[idx0][-1] + dandrogramRecord[idx1][3] * dandrogramRecord[idx1][-1]) / cluster_eles
    # dis = np.linalg.norm(dandrogramRecord[idx0][-1] - dandrogramRecord[idx1][-1])
    # if (round(np.sqrt(ward_dis_square(dandrogramRecord, idx0, idx1)), 5) != round(cluster_mat[i][2], 5)):
    #     print(np.sqrt(ward_dis_square(dandrogramRecord, idx0, idx1)), cluster_mat[i][2])
    dandrogramRecord.append([idx0, idx1, cluster_eles, cluster_centroid0])


with open(processed_filepath + '/init_data_voxel_image_tsne.csv', 'w', newline='') as fc:
  writer = csv.writer(fc)
  writer.writerows(data_embedding)






{
    # 'icoord': [[5.0, 5.0, 15.0, 15.0], [25.0, 25.0, 35.0, 35.0], [10.0, 10.0, 30.0, 30.0], [45.0, 45.0, 55.0, 55.0], 
    #         [65.0, 65.0, 75.0, 75.0], [50.0, 50.0, 70.0, 70.0], [105.0, 105.0, 115.0, 115.0], [95.0, 95.0, 110.0, 110.0], 
    #         [85.0, 85.0, 102.5, 102.5], [60.0, 60.0, 93.75, 93.75], [20.0, 20.0, 76.875, 76.875]], 
    # 'dcoord': [[0.0, 0.0, 0.0, 0.0], [0.0, 15.678327139954764, 15.678327139954764, 0.0], [0.0, 27.256331439449397, 27.256331439449397, 15.678327139954764], 
    #         [0.0, 0.0, 0.0, 0.0], [0.0, 9.624373862842635, 9.624373862842635, 0.0], [0.0, 15.079784119478944, 15.079784119478944, 9.624373862842635], 
    #         [0.0, 0.0, 0.0, 0.0], [0.0, 14.83561757278, 14.83561757278, 0.0], [0.0, 15.844805348733685, 15.844805348733685, 14.83561757278], 
    #         [15.079784119478944, 46.550743072231654, 46.550743072231654, 15.844805348733685], [27.256331439449397, 47.31132268844892, 47.31132268844892, 46.550743072231654]], 
    # 'ivl': ['cabinet', 'cabinet', 'cabinet', 'cabinet', 'bench', 'bench', 'bench', 'bench', 'plane', 'plane', 'plane', 'plane'], 
    'leaves': [8, 11, 9, 10, 5, 7, 4, 6, 0, 1, 2, 3], 
    # 'color_list': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3', 'C0', 'C0'], 
    # 'leaves_color_list': ['C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2', 'C3', 'C3', 'C3', 'C3']
}

# [[ 2.          3.          0.          2.        ]
#  [ 5.          7.          0.          2.        ]
#  [ 8.         11.          0.          2.        ]
#  [ 4.          6.          9.62437386  2.        ]
#  [ 1.         12.         14.83561757  3.        ]
#  [13.         15.         15.07978412  4.        ]
#  [ 9.         10.         15.67832714  2.        ]
#  [ 0.         16.         15.84480535  4.        ]
#  [14.         18.         27.25633144  4.        ]
#  [17.         19.         46.55074307  8.        ]
#  [20.         21.         47.31132269 12.        ]]

# 我现在有一个层次聚类，每列含义：
# 1, 2: 该簇包含的两个下属簇序号（如果是叶节点就是叶元素在输入列表中的序号），新聚合的簇序号即输入列表长度+簇在层次聚类列表中的序号
# 3: 聚合的两个簇的距离
# 4: 该簇包含的数量
# 前端做成显示离体素相似度最近的簇的各个子节点（展开两次也就是显示四个？），同时把展示簇下的第一个叶节点发送给前端，同时显示三个阴影表示另外三个子节点



# with open(processed_filepath + '/' + 'bench 0', 'r', encoding='utf-8') as processed_data:
#     # 体素，图片
#     data = json.load(processed_data)
#     print(type(data['voxel']))
#     print(type(data['image']))
