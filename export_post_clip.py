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

###################################### Text Queries ###############################################

id_to_label = {'02691156': 0, '02828884': 1, '02933112': 2, '02958343': 3, '03001627': 4, '03211117': 5, '03636649': 6, '03691459': 7, '04090263': 8, '04256520': 9, '04379243': 10, '04401088': 11, '04530566': 12}

label_to_category = {0: 'airplane', 1:'bench', 2:'cabinet', 3:'car',  4:'chair', 5:'monitor', 6:'lamp',  7:'loudspeaker',  8:'gun',  9:'sofa',  10:'table',  11:'phone',  12:'boat'}


id_to_sub_category = {

"02691156": ["airplane", "jet", "fighter plane", "biplane", "seaplane", "space shuttle", "supersonic plane", "rocket plane", "delta wing", "swept wing plane" , "straight wing plane", "propeller plane"],     

"02828884": ["bench", "pew", "flat bench", "settle", "back bench", "laboratory bench", "storage bench"],    

"02933112": ["cabinet", "garage cabinet", "desk cabinet"]  ,
    
"02958343": ["car", "bus", "shuttle-bus", "pickup car", "truck", "suv", "sports car", "limo", "jeep", "van", "gas guzzler", "race car", "monster truck", "armored", "atv", "microbus", "muscle car", "retro car", "wagon car", "hatchback", "sedan", "ambulance", "roadster car", "beach wagon"],

"03001627": ["chair", "arm chair",  "bowl chair", "rocking chair", "egg chair", "swivel chair", "bar stool", "ladder back chair", "throne", "office chair", "wheelchair", "stool", "barber chair", "folding chair", "lounge chair", "vertical back chair", "recliner", "wing chair", "sling"],  

"03211117": ["monitor", "crt monitor"],

"03636649": ["lamp",  "street lamp", "fluorescent lamp", "gas lamp", "bulb"],
    
"03691459": ["loudspeaker", "subwoofer speaker"],
    
"04090263": ["gun", "machine gun", "sniper rifle", "pistol", "shotgun"],    
 
"04256520": ["sofa", "double couch",  "love seat", "chesterfield", "convertiable sofa", "L shaped sofa", "settee sofa", "daybed", "sofa bed", "ottoman"],

"04379243": ["table", "dressing table", "desk", "refactory table",  "counter", "operating table", "stand", "billiard table", "pool table", "ping-pong table", "console table"],

 "04401088": ["phone", "desk phone", "flip-phone"],
   
"04530566": ["boat", "war ship", "sail boat", "speedboat", "cabin cruiser",  "yacht"],
    
}

id_to_shape_attribute = {

"02691156": ["triangular"],     

"02828884": ["square", "round", "circular", "rectangular", "thick", "thin"],    

"02933112": ["cuboid", "round", "rectangular", "thick", "thin"]  ,
    
"02958343": ["square", "round", "rectangular", "thick", "thin"],

"03001627": ["square", "round", "rectangular", "thick", "thin"],  

"03211117": ["square", "round", "rectangular", "thick", "thin"],

"03636649": ["square", "round", "rectangular", "cuboid", "circular", "thick", "thin"],
    
"03691459": ["square", "round", "rectangular", "circular", "thick", "thin"],
    
"04090263": ["thick", "thin"],    
 
"04256520": ["square", "round", "rectangular", "thick", "thin"],

"04379243": ["square", "round", "circular", "rectangular", "thick", "thin"],

"04401088": ["square", "rectangular", "thick", "thin"],
   
"04530566": ["square", "round", "rectangular", "thick", "thin"],
    
}

id_to_other_stuff = {

"02691156": ["boeing", "airbus", "f-16", "plane", "aeroplane", "aircraft", "commerical plane"],     

"02828884": ["park bench"],    

"02933112": ["dresser", "cupboard", "container", "case", "locker", "cupboard", "closet", "sideboard"]  ,
    
"02958343": ["auto", "automobile", "motor car"],

"03001627": ["seat", "cathedra"],  

"03211117": ["TV", "digital display", "flat panel display", "screen", "television", "telly", "video"],

"03636649": ["lantern", "table lamp", "torch"],
    
"03691459": ["speaker", "speaker unit", "tannoy"],
    
"04090263": ["ak-47", "uzi", "M1 Garand", "M-16","firearm", "shooter", "weapon"],    
 
"04256520": ["couch", "lounge", "divan", "futon"],

"04379243": ["altar table", "worktop", "workbench"],

"04401088": ["telephone",  "telephone set", "cellular telephone", "cellular phone", "cellphone", "cell", "mobile phone", "iphone"],
   
"04530566": ["rowing boat", "watercraft", "ship", "canal boat", "ferry", "steamboat", "barge"],
    
}


###################################### Text Queries ###############################################


def save_voxel_images_lerp_2(net, latent_flow_model, clip_model, args, total_text_query, save_path, resolution=64, num_figs_per_query=5):
    net.eval()
    latent_flow_model.eval()
    clip_model.eval()
    count = 1
    num_figs = num_figs_per_query
    with torch.no_grad():
        voxel_size = resolution
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())
        assert(len(total_text_query) == 2)
        embs = []
        texts = []
        for text_in in tqdm(total_text_query):
            # print(text_in)
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
            embs.append(decoder_embs)
            texts.append(text_in)

        inter = 0.0
        for idx in range(51):
            out = net.decoding(torch.lerp(embs[0], embs[1], inter), query_points)
            voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
            
            voxel_num = 0
            for voxel_in in voxels_out:
                out_file = os.path.join(save_path, texts[0] + "_" + texts[1] + "_" + str(voxel_num) + "_" + str(idx) + ".png")
                voxel_save(voxel_in, None, out_file=out_file)
                voxel_num = voxel_num + 1
            inter += 0.02

def gen_embeddings(latent_flow_model, clip_model, args, total_text_query, save_path, num_figs_per_query=5):
    latent_flow_model.eval()
    clip_model.eval()
    count = 1
    num_figs = num_figs_per_query
    with torch.no_grad():
        shape_embs = []
        shape_embs_torch = []
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
            shape_embs.append(decoder_embs.detach().cpu().numpy())
            shape_embs_torch.append(decoder_embs)
        np.save(save_path + '/shape_embs.npy', np.array(shape_embs))
        return shape_embs_torch

def embs2voxel_lerp4(net, args, total_embs, save_path, xval, yval, resolution=64, num_figs_per_query=5):
    # total_embs的长度应小于4, index0-3分别代表: 左下, 右下, 左上, 右上
    # xval, yval∈[0, 1], 代表横向和纵向插值
    net.eval()
    num_figs = num_figs_per_query
    with torch.no_grad():
        voxel_size = resolution
        shape = (voxel_size, voxel_size, voxel_size)
        p = visualization.make_3d_grid([-0.5] * 3, [+0.5] * 3, shape).type(torch.FloatTensor).to(args.device)
        query_points = p.expand(num_figs, *p.size())

        # 这里暂时先用最简单的三次线性插值，即上下分别根据xval插值，然后上下合并yval插值
        res_emb = torch.lerp(torch.lerp(total_embs[0], total_embs[1], xval), torch.lerp(total_embs[2], total_embs[3], xval), yval)
        out = net.decoding(res_emb, query_points)
        voxels_out = (out.view(num_figs, voxel_size, voxel_size, voxel_size) > args.threshold).detach().cpu().numpy()
        
        for voxel_in in voxels_out:
            out_file = os.path.join(save_path, str(xval) + "_" + str(yval)+ ".png")
            voxel_save(voxel_in, None, out_file=out_file)

##################################### Main and Parser stuff #################################################3

def get_local_parser_test(mode="args"):
    parser = get_local_parser(mode="parser")
    parser.add_argument("--experiment_mode",  type=str, default='save_voxel_on_query', metavar='N', help='experiment type')
    parser.add_argument("--classifier_checkpoint",  type=str, default="./exps/classifier/checkpoints/best.pt", metavar='N', help='what is the classifier checkpoint for FID, Acc and Stuff')
    parser.add_argument("--checkpoint_nf",  type=str, default="best", metavar='N', help='what is the checkpoint for nf')
    parser.add_argument("--prefix",  type=str, default="a", metavar='N', help='add or remove')
    parser.add_argument("--post_dataset",  type=str, default=None, help='if want to use diff dataset during post')
    parser.add_argument("--checkpoint_dir_base",  type=str, default=None, help='Checkpoint directory for autoencoder')
    parser.add_argument("--output_dir",  type=str, default="./exps/output_dir", help='output dir')
    parser.add_argument("--checkpoint_dir_prior",  type=str, default=None, help='Checkpoint for prior')

    args = parser.parse_args()
    
    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser

def export_voxel():
    args = get_local_parser_test() 
    print("################################################")
    # for item in args:
    #     print (item)
    print (args)
    print("################################################")
    ### Directories for generating stuff and logs    cls_cal_category
    test_log_filename = osp.join(args.output_dir, 'test_log.txt')
    helper.create_dir(args.output_dir)
    helper.setup_logging(test_log_filename, args.log_level, 'w')
    args.query_generate_dir =  osp.join(args.output_dir, 'query_generate_dir') + "/"                     
    helper.create_dir(args.query_generate_dir)
    args.vis_gen_dir =  osp.join(args.output_dir, 'vis_gen_dir') + "/"                  # 保存生成图像         
    helper.create_dir(args.vis_gen_dir)

    manualSeed = args.seed_nf
    helper.set_seed(manualSeed)

    ### Dataloader stuff 
    if args.experiment_mode not in ["save_voxel_on_query", "cls_cal_single", "cls_cal_category", "gen_embeddings_on_query"]:
        logging.info("#############################")
        train_dataloader, total_shapes  = get_dataloader(args, split="train")
        args.total_shapes = total_shapes
        logging.info("Train Dataset size: {}".format(total_shapes))
        val_dataloader, total_shapes_val  = get_dataloader(args, split="val")
        logging.info("Val Dataset size: {}".format(total_shapes_val))
        test_dataloader, total_shapes_test, test_dataset  = get_dataloader(args, split="test", dataset_flag=True)
        logging.info("Test Dataset size: {}".format(total_shapes_test))
        logging.info("#############################")

    device, gpu_array = helper.get_device(args)
    args.device = device     

    ### Network stuff 

    net = autoencoder.get_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt", map_location=args.device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    logging.info("Loaded the autoencoder: {}".format(args.checkpoint_dir_base +"/"+ args.checkpoint +".pt"))
    
    args, clip_model = get_clip_model(args)

    latent_flow_network = latent_flows.get_generator(args.emb_dims, args.cond_emb_dim, device, flow_type=args.flow_type, num_blocks=args.num_blocks, num_hidden=args.num_hidden)
    
    checkpoint_nf_path = os.path.join(args.checkpoint_dir_prior,  args.checkpoint_nf +".pt")
    logging.info("Loaded the nf model: {}".format(checkpoint_nf_path))
    
    checkpoint = torch.load(checkpoint_nf_path, map_location=args.device)
    latent_flow_network.load_state_dict(checkpoint['model'])
    latent_flow_network.eval()
 
    if args.experiment_mode == "save_voxel_on_query":
        save_path = args.vis_gen_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        torch.multiprocessing.set_sharing_strategy('file_system')
        if args.text_query is None:
            logging.info("Please add text query using text_query args argument")
        else: 
            save_voxel_images(net, latent_flow_network, clip_model, args, args.text_query, save_path, resolution=64, num_figs_per_query=1)
            # save_voxel_images_lerp_2(net, latent_flow_network, clip_model, args, args.text_query, save_path, resolution=64, num_figs_per_query=1)
    elif args.experiment_mode == "gen_embeddings_on_query":
        save_path = args.vis_gen_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        torch.multiprocessing.set_sharing_strategy('file_system')
        embs_gen = []
        if args.text_query is None:
            logging.info("Please add text query using text_query args argument")
        else: 
            embs_gen = gen_embeddings(latent_flow_network, clip_model, args, args.text_query, args.output_dir, num_figs_per_query=1)
        if (len(embs_gen) == 4):
            while (True):
                xval = float(input("xval: "))
                yval = float(input("yval: "))
                if (xval < 0 or yval < 0):
                    break
                embs2voxel_lerp4(net, args, embs_gen, save_path, xval, yval, resolution=64, num_figs_per_query=1)