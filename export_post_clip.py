import os
import os.path as osp
import sys
import logging

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data import Dataset, DataLoader


from utils import helper
from utils import visualization
from utils import experimenter

from train_autoencoder import experiment_name, parsing
from train_post_clip import get_dataloader, experiment_name2, get_condition_embeddings, get_local_parser, get_clip_model
from dataset import shapenet_dataset
from networks import autoencoder, latent_flows

import clip

from flask import Flask, jsonify, request, render_template
from werkzeug.routing import BaseConverter

app = Flask(__name__)


class RegexConverter(BaseConverter):
    def __init__(self, url_map, *args):
        super(RegexConverter, self).__init__(url_map)
        
        # 将接收的第1个参数当作匹配规则进行保存
        self.regex = args[0]

app.url_map.converter['re'] = RegexConverter


def get_local_parser_test(mode="args"):
    parser = get_local_parser(mode="parser")
    parser.add_argument("--experiment_mode",  type=str, default='save_voxel_on_query', metavar='N', help='experiment type')
    parser.add_argument("--classifier_checkpoint",  type=str, default="./exps/classifier/checkpoints/best.pt", metavar='N', help='what is the classifier checkpoint for FID, Acc and Stuff')
    parser.add_argument("--checkpoint_nf",  type=str, default="best", metavar='N', help='what is the checkpoint for nf')
    parser.add_argument("--prefix",  type=str, default="a", metavar='N', help='add or remove')
    parser.add_argument("--post_dataset",  type=str, default=None, help='if want to use diff dataset during post')
    parser.add_argument("--checkpoint_dir_base",  type=str, default=None, help='Checkpoint directory for autoencoder')
    parser.add_argument("--checkpoint_dir_prior",  type=str, default=None, help='Checkpoint for prior')

    args = parser.parse_args()
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

args = get_local_parser_test() 
args.checkpoint_dir_base="./exps/models/autoencoder"
args.checkpoint='best_iou'
args.checkpoint_nf='best'
args.checkpoint_dir_prior="./exps/models/prior"
args.threshold=0.1

manualSeed = args.seed_nf
helper.set_seed(manualSeed)

device, gpu_array = helper.get_device(args)
args.device = device     

# net = None
# latent_flow_model = None
# clip_model = None


# def save_voxel_images_lerp_2(net, latent_flow_model, clip_model, args, total_text_query, save_path, resolution=64, num_figs_per_query=5):
#     net.eval()
#     latent_flow_model.eval()
#     clip_model.eval()
#     num_figs = num_figs_per_query
#     with torch.no_grad():
#         voxel_size = resolution
#         shape = (voxel_size, voxel_size, voxel_size)
#         p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
#         query_points = p.expand(num_figs, *p.size())
#         assert(len(total_text_query) == 2)
#         embs = []
#         texts = []
#         for text_in in tqdm(total_text_query):
#             # print(text_in)
#             ##########
#             text = clip.tokenize([text_in]).to(args.device)
#             text_features = clip_model.encode_text(text)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             ###########
#             torch.manual_seed(5)
#             mean_shape = torch.zeros(1, args.emb_dims).to(args.device) 
#             noise = torch.Tensor(num_figs-1, args.emb_dims).normal_().to(args.device) 
#             noise = torch.clip(noise, min=-1, max=1)
#             noise = torch.cat([mean_shape, noise], dim=0)
#             decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))
#             embs.append(decoder_embs)
#             texts.append(text_in)

#         inter = 0.0
#         for idx in range(51):
#             out = net.decoding(torch.lerp(embs[0], embs[1], inter), query_points)
#             voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            
#             voxel_num = 0
#             for voxel_in in voxels_out:
#                 out_file = os.path.join(save_path, texts[0] + "_" + texts[1] + "_" + str(voxel_num) + "_" + str(idx) + ".png")
#                 voxel_save(voxel_in, None, out_file=out_file)
#                 voxel_num = voxel_num + 1
#             inter += 0.02

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/get_embeddings_by_text_query', methods=['GET', 'POST'])
def get_embeddings_by_text_query():
    total_text_query = request.json
    global shape_embs_torch
    shape_embs_torch = []
    shape_embs = []
    clip_model.eval()
    latent_flow_model.eval()
    if (total_text_query != None):
        print(total_text_query)
        with torch.no_grad():
            num_figs = 1
            for text_in in tqdm(total_text_query):
                ##########
                text = clip.tokenize([text_in]).to(args.device)
                text_features = clip_model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                ###########
                torch.manual_seed(5)
                mean_shape = torch.zeros(1, args.emb_dims).to(args.device) 
                noise = torch.Tensor(num_figs-1, args.emb_dims).normal_().to(args.device) 
                noise = torch.clip(noise, min=-1, max=1)
                noise = torch.cat([mean_shape, noise], dim=0)
                decoder_embs = latent_flow_model.sample(num_figs, noise=noise, cond_inputs=text_features.repeat(num_figs,1))
                shape_embs.append(decoder_embs.detach().cpu().numpy().tolist()[0])
                shape_embs_torch.append(decoder_embs)
    else:
        print("no query")
    print (len(shape_embs_torch))
    return jsonify(shape_embs)

# ind0-3分别代表: 左下, 右下, 左上, 右上
@app.route('/get_voxel/<int:idx0>-<re("-?[0-9]+"):idx1>-<re("-?[0-9]+"):idx2>-<re("-?[0-9]+"):idx3>/<float:xval>-<float:yval>', methods=['GET', 'POST'])
def get_voxel_interpolation(idx0, idx1, idx2, idx3, xval, yval):
    print (idx0, idx1, idx2, idx3)
    net.eval()
    num_figs = 1
    resolution = 64
    with torch.no_grad():
        voxel_size = resolution
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())

        res_emb = shape_embs_torch[0]
        
        if (idx1 == -1):    # 1个embedding无插值
            pass
        elif (idx2 == -1):  # 2个embedding 一次线性插值
            res_emb = torch.lerp(shape_embs_torch[idx0], shape_embs_torch[idx1], xval)
        elif (idx3 == -1):  # 3个embedding 三角形重心坐标插值
            res_emb = xval * shape_embs_torch[idx1] + yval * shape_embs_torch[idx2] + (1.0 - xval - yval) * shape_embs_torch[idx0]
        else:               # 4个embedding 三次线性插值
            res_emb = torch.lerp(torch.lerp(shape_embs_torch[idx0], shape_embs_torch[idx1], xval), torch.lerp(shape_embs_torch[idx2], shape_embs_torch[idx3], xval), yval)
        out = net.decoding(res_emb, query_points)
        voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
    # return jsonify([90])
    return jsonify(voxels_out[0].tolist())


# def embs2voxel_lerp4(net, args, total_embs, save_path, xval, yval, resolution=64, num_figs_per_query=5):
#     # total_embs的长度应小于4, index0-3分别代表: 左下, 右下, 左上, 右上
#     # xval, yval∈[0, 1], 代表横向和纵向插值
#     net.eval()
#     num_figs = num_figs_per_query
#     with torch.no_grad():
#         voxel_size = resolution
#         shape = (voxel_size, voxel_size, voxel_size)
#         p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
#         query_points = p.expand(num_figs, *p.size())

#         # 这里暂时先用最简单的三次线性插值，即上下分别根据xval插值，然后上下合并yval插值
#         res_emb = torch.lerp(torch.lerp(total_embs[0], total_embs[1], xval), torch.lerp(total_embs[2], total_embs[3], xval), yval)
#         out = net.decoding(res_emb, query_points)
#         voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
        
#         for voxel_in in voxels_out:
#             # out_file = os.path.join(save_path, str(xval) + "_" + str(yval)+ ".png")
#             # voxel_save(voxel_in, None, out_file=out_file)
#             return voxel_in

##################################### Main and Parser stuff #################################################3





if __name__ == '__main__':
    ### Network stuff 
    global net
    global latent_flow_model
    global clip_model
    global shape_embs_torch
    shape_embs_torch = []
    net = autoencoder.get_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    logging.info("Loaded the autoencoder: {}".format(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt"))

    args, clip_model = get_clip_model(args)

    latent_flow_model = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)

    checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
    logging.info("Loaded the nf model: {}".format(checkpoint_nf_path))

    checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
    latent_flow_model.load_state_dict(checkpoint['model'])
    latent_flow_model.eval()
    
 
    # if args.experiment_mode == "save_voxel_on_query":
    #     save_path = args.vis_gen_dir
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path) 
    #     torch.multiprocessing.set_sharing_strategy('file_system')
    #     if args.text_query is None:
    #         logging.info("Please add text query using text_query args argument")
    #     else: 
    #         save_voxel_images(net, latent_flow_network, clip_model, args, args.text_query, save_path, resolution=64, num_figs_per_query=1)
    #         # save_voxel_images_lerp_2(net, latent_flow_network, clip_model, args, args.text_query, save_path, resolution=64, num_figs_per_query=1)
    # elif args.experiment_mode == "gen_embeddings_on_query":
    #     save_path = args.vis_gen_dir
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path) 
    #     torch.multiprocessing.set_sharing_strategy('file_system')
    #     embs_gen = []
    #     if args.text_query is None:
    #         logging.info("Please add text query using text_query args argument")
    #     else: 
    #         embs_gen = gen_embeddings(latent_flow_network, clip_model, args, args.text_query, args.output_dir, num_figs_per_query=1)
    #     if (len(embs_gen) == 4):
    #         while (True):
    #             xval = float(input("xval: "))
    #             yval = float(input("yval: "))
    #             if (xval < 0 or yval < 0):
    #                 break
    #             embs2voxel_lerp4(net, args, embs_gen, save_path, xval, yval, resolution=64, num_figs_per_query=1)
    app.debug = True
    app.run()