from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
#from concurrent.futures import ProcessPoolExecutor as Pool
import time
import torch
import numpy
import test

from pruneshift.networks import create_network
from pruneshift.prune_info import PruneInfo
from pruneshift.utils import get_model_complexity_prune
import torch.nn as nn
from torch.autograd import Variable
import numpy
from pruneshift.datamodules import ShiftDataModule as dm

IMGR_TAG = "ImageNet100-R Error"
IMGC_TAG = "ImageNet100-C Error"
IMG_TAG = "ImageNet100 Error"

#path="/misc/scratchSSD2/datasets/ILSVRC2012-100/"
path="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/"
tester = dm(name='imagenet', root=path)
tester.setup(stage='test')
test_loader=tester.test_dataloader()

def calculate_masked_layers(network):
    l1_total_mask=0
    l1_non_zero=0
    l2_total_mask=0
    l2_non_zero=0
    l3_total_mask=0
    l3_non_zero=0
    l4_total_mask=0
    l4_non_zero=0
    for a, p in network.named_buffers():
        if 'weight_mask' in a:
            a_copy=p.detach().cpu().numpy()
            if 'layer1' in a:
                l1_total_mask+=len(a_copy.flatten())
                l1_non_zero+=numpy.count_nonzero(a_copy)
            if 'layer2' in a:
                l2_total_mask+=len(a_copy.flatten())
                l2_non_zero+=numpy.count_nonzero(a_copy)
            if 'layer3' in a:
                l3_total_mask+=len(a_copy.flatten())
                l3_non_zero+=numpy.count_nonzero(a_copy)
            if 'layer4' in a:
                l4_total_mask+=len(a_copy.flatten())
                l4_non_zero+=numpy.count_nonzero(a_copy)
    try:
        l1_remaining = l1_non_zero/l1_total_mask
    except:
        l1_remaining = 1.0
        
    try: 
        l2_remaining = l2_non_zero/l2_total_mask
    except:
        l2_remaining = 1.0
    try:
        l3_remaining = l3_non_zero/l3_total_mask
    except:
        l3_remaining = 1.0
    try:
        l4_remaining = l4_non_zero/l4_total_mask  
    except:
        l4_remaining = 1.0
    return l1_remaining, l2_remaining, l3_remaining, l4_remaining


def to_percentage(df):
    for column in df.columns:
        if not "Error" in column:
            continue
        df[column] = df[column] * 100 

def find_folders(*paths):
    exp_paths = []
    for path in paths:
        exp_paths.extend(Path(path).glob("**/metrics.csv"))
        
    return [p.parent for p in exp_paths]

def calculate_info(row):
    path = Path(row["Path"]) / "checkpoint/last.ckpt"
    #path = Path(str(path).replace("/work/dlclarge1/", "/misc/lmbraid19/agnihotr/"))
    if not path.exists() and not row["ensemble"]:
        print(f"Did not find {path}.")
        return -1, -1
    #if row["ensemble"]:
        #path=None

    #row["network1"]=correct_path(row["network1"])
    #row["network2"]=correct_path(row["network2"])
    #row["network3"]=correct_path(row["network3"])
    
    
    net = create_network("imagenet", row["Network"], 100, 
                         ckpt_path=path, 
                         scaling_factor=row["Scaling"],
                        ensemble=row["ensemble"],
                        loading_ensemble=row["loading_ensemble"],
                        multiheaded=row["multiheaded"],
                        network1_path=row["network1"],
                        network2_path=row["network2"],
                        network3_path=row["network3"])#imagenet
    size = PruneInfo(net).network_size()
    #size = calculate_size(net)
    macs, _ = get_model_complexity_prune(net, input_res=(3, 64, 256))
    # print(macs)
    i =0
    tensor=torch.randn(1, 3, 3, 3)
    start_time=0;
    end_time=0;
    elapsed_time=0;
    while(i<110):
        if i>9:
            start_time=time.time()
        #net(tensor)
        if i>9:
            end_time=time.time()
        elapsed_time+=end_time - start_time
        i+=1
            
    del net
    return macs, size / 1e6

def correct_path(path):
    if path is not None:
        path = path.replace("/work/dlclarge1/", "/misc/lmbraid19/agnihotr/")
    return path

def add_ece(row):
    path = Path(row["Path"]) / "checkpoint/last.ckpt"
    #path = Path(str(path).replace("/work/dlclarge1/", "/misc/lmbraid19/agnihotr/"))
    if not path.exists() and not row["ensemble"]:
        print(f"Did not find {path}.")
        return -1, -1
    #if row["ensemble"]:
        #path=None
    #row["network1"]=correct_path(row["network1"])
    #row["network2"]=correct_path(row["network2"])
    #row["network3"]=correct_path(row["network3"])
    
    net = create_network("imagenet", row["Network"], 100, 
                         ckpt_path=path, 
                         scaling_factor=row["Scaling"],
                        ensemble=row["ensemble"],
                        loading_ensemble=row["loading_ensemble"],
                        multiheaded=row["multiheaded"],
                        network1_path=row["network1"],
                        network2_path=row["network2"],
                        network3_path=row["network3"])#imagenet
    
    
    
    net=nn.DataParallel(net)
    mean_ece, mean_err = test.calculate_ece_corr(net, row["multiheaded"])
    mean_flip_prob, mean_ece_per, mean_err_per = test.calculate_on_per(net, row["multiheaded"])
    ece, err = test.calculate_ece(net, row["multiheaded"])
    
    
    perf = test.top_perf(net, row["multiheaded"])
    l1, l2, l3, l4 = calculate_masked_layers(net)
    return ece.item(), err.item(), l1, l2, l3, l4, perf, mean_ece.item(), mean_err.item(), mean_flip_prob, mean_ece_per.item(), mean_err_per.item()

def add_info(df, num_workers=8):
    rows_iter = (row for _, row in df[["Path", "Scaling", "Network", "ensemble", "loading_ensemble", "multiheaded", "network1", "network2", "network3"]].iterrows())
    
    with Pool(num_workers) as pool:
        df[["MACs", "Model Size"]] = pool.map(calculate_info, rows_iter)
    ece = []
    err = []
    mean_ece = []
    mean_err = []
    mean_flip_prob, mean_ece_per, mean_err_per = [],[],[]
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    
    p_test_1 =[]
    class_test_1=[]
    p_test_2 =[]
    class_test_2=[]
    p_test_3 =[]
    class_test_3=[]
    p_target_test=[]
    class_target_test=[]
    
    p_b1_1 =[]
    class_b1_1=[]
    p_b1_2 =[]
    class_b1_2=[]
    p_b1_3 =[]
    class_b1_3=[]
    p_target_b1=[]
    class_target_b1=[]
    
    p_b3_1 =[]
    class_b3_1=[]
    p_b3_2 =[]
    class_b3_2=[]
    p_b3_3 =[]
    class_b3_3=[]
    p_target_b3=[]
    class_target_b3=[]
    
    p_b5_1 =[]
    class_b5_1=[]
    p_b5_2 =[]
    class_b5_2=[]
    p_b5_3 =[]
    class_b5_3=[]
    p_target_b5=[]
    class_target_b5=[]
    
    p_r_1 =[]
    class_r_1=[]
    p_r_2 =[]
    class_r_2=[]
    p_r_3 =[]
    class_r_3=[]
    p_target_r=[]
    class_target_r=[]
    
    for _, row in df[["Path", "Scaling", "Network", "ensemble", "loading_ensemble", "multiheaded", "network1", "network2", "network3"]].iterrows():
        #print(row)
        ece_info, err_info, l1_info, l2_info, l3_info, l4_info, perf, mean_ece_info, mean_err_info, mean_flip_prob_info, mean_ece_per_info, mean_err_per_info = add_ece(row)
        ece.append(ece_info)
        err.append(err_info)
        mean_ece.append(mean_ece_info)
        mean_err.append(mean_err_info)
        mean_flip_prob.append(mean_flip_prob_info)
        mean_ece_per.append(mean_ece_per_info)
        mean_err_per.append(mean_err_per_info)
        l1.append(l1_info)
        l2.append(l2_info)
        l3.append(l3_info)
        l4.append(l4_info)
        
        p_test_1.append(perf[0][0].item())
        p_test_2.append(perf[0][1].item())
        p_test_3.append(perf[0][2].item())
        class_test_1.append(perf[1][0])
        class_test_2.append(perf[1][1])
        class_test_3.append(perf[1][2])
        p_target_test.append(perf[2])
        class_target_test.append(perf[3])
        
        p_b1_1.append(perf[4][0].item())
        p_b1_2.append(perf[4][1].item())
        p_b1_3.append(perf[4][2].item())
        class_b1_1.append(perf[5][0])
        class_b1_2.append(perf[5][1])
        class_b1_3.append(perf[5][2])
        p_target_b1.append(perf[6])
        class_target_b1.append(perf[7])
        
        p_b3_1.append(perf[8][0].item())
        p_b3_2.append(perf[8][1].item())
        p_b3_3.append(perf[8][2].item())
        class_b3_1.append(perf[9][0])
        class_b3_2.append(perf[9][1])
        class_b3_3.append(perf[9][2])
        p_target_b3.append(perf[10])
        class_target_b3.append(perf[11])
        
        p_b5_1.append(perf[12][0].item())
        p_b5_2.append(perf[12][1].item())
        p_b5_3.append(perf[12][2].item())
        class_b5_1.append(perf[13][0])
        class_b5_2.append(perf[13][1])
        class_b5_3.append(perf[13][2])
        p_target_b5.append(perf[14])
        class_target_b5.append(perf[15])
        
        p_r_1.append(perf[16][0].item())
        p_r_2.append(perf[16][1].item())
        p_r_3.append(perf[16][2].item())
        class_r_1.append(perf[17][0])
        class_r_2.append(perf[17][1])
        class_r_3.append(perf[17][2])
        p_target_r.append(perf[18])
        class_target_r.append(perf[19])
        
        
    df["ECE"]=ece
    df["err"]=err
    df["mean ECE corr"]=mean_ece
    df["mCE calc"] = mean_err
    df["mFP"] = mean_flip_prob
    df["mean ECE pertubation"]= mean_ece_per
    df["mean error pertubation"] = mean_err_per
    df["Layer1 Remaining"]=l1
    df["Layer2 Remaining"]=l2
    df["Layer3 Remaining"]=l3
    df["Layer4 Remaining"]=l4
    
    df["Test Class 1 Probability"]=p_test_1 
    df["Test Class 1 Name"]=class_test_1
    df["Test Class 2 Probability"]=p_test_2 
    df["Test Class 2 Name"]=class_test_2
    df["Test Class 3 Probability"]=p_test_3 
    df["Test Class 3 Name"]=class_test_3
    df["Test Target Probability"]=p_target_test
    df["Test Target Name"]=class_target_test
    
    df["Bright 1 Class 1 Probability"]=p_b1_1 
    df["Bright 1 Class 1 Name"]=class_b1_1
    df["Bright 1 Class 2 Probability"]=p_b1_2 
    df["Bright 1 Class 2 Name"]=class_b1_2
    df["Bright 1 Class 3 Probability"]=p_b1_3 
    df["Bright 1 Class 3 Name"]=class_b1_3
    df["Bright 1 Target Probability"]=p_target_b1
    df["Bright 1 Target Name"]=class_target_b1
    
    
    df["Bright 3 Class 1 Probability"]=p_b3_1 
    df["Bright 3 Class 1 Name"]=class_b3_1
    df["Bright 3 Class 2 Probability"]=p_b3_2 
    df["Bright 3 Class 2 Name"]=class_b3_2
    df["Bright 3 Class 3 Probability"]=p_b3_3 
    df["Bright 3 Class 3 Name"]=class_b3_3
    df["Bright 3 Target Probability"]=p_target_b3
    df["Bright 3 Target Name"]=class_target_b3
    
    
    df["Bright 5 Class 1 Probability"]=p_b5_1 
    df["Bright 5 Class 1 Name"]=class_b5_1
    df["Bright 5 Class 2 Probability"]=p_b5_2 
    df["Bright 5 Class 2 Name"]=class_b5_2
    df["Bright 5 Class 3 Probability"]=p_b5_3 
    df["Bright 5 Class 3 Name"]=class_b5_3
    df["Bright 5 Target Probability"]=p_target_b5
    df["Bright 5 Target Name"]=class_target_b5
    
    df["Rend Class 1 Probability"]=p_r_1 
    df["Rend Class 1 Name"]=class_r_1
    df["Rend Class 2 Probability"]=p_r_2 
    df["Rend Class 2 Name"]=class_r_2
    df["Rend Class 3 Probability"]=p_r_3 
    df["Rend Class 3 Name"]=class_r_3
    df["Rend Target Probability"]=p_target_r
    df["Rend Target Name"]=class_target_r
    
    

    return df

def teacher_name(conf):
    #print(conf.teacher.ckpt_path)
    if conf.teacher.name == "swsl_resnet50":
        return "SWSL ResNet50"    
    elif conf.teacher.name == "resnet50":
        path = conf.teacher.ckpt_path
        if path is None:
            return "STD ResNet50"
        elif "img100/amda_resnet50" in path:
            return "Img100-AMDA_ResNet50"
        elif "augmix" in path and "deepaugment" in path:
            return "AMDA ResNet50"
        elif "augmix" in path:
            return "AM ResNet50"
        elif "deepaugment" in path:
            return "DA ResNet50"
    elif conf.teacher.ckpt_path is None:
        return "STD_"+conf.teacher.name
    elif "deepaugment" in conf.teacher.ckpt_path:
            return "AMDA_"+conf.teacher.name
    elif 'img100/amda' in conf.teacher.ckpt_path:
        return "Img100-AMDA_"+conf.teacher.name
    elif 'resnet34/standard' in conf.teacher.ckpt_path:
        return "Img100-STD_"+conf.teacher.name
    elif 'resnet34/amda' in conf.teacher.ckpt_path:
        return "Img100-AMDA_"+conf.teacher.name
    
    return conf.teacher.name
    return "custom"
    raise NotImplementedError

def prune_method(conf):
    """ Returns the name and the amount."""
    if conf.prune.method == "l1_channels":
        return "L¹ Filter"
    elif conf.prune.method == "global_weight":
        return "Global Weight"
    elif conf.prune.method == "layer_weight":
        return "Layer Weight"
    elif conf.prune.method == "random_channels":
        return "Random Filter"
    elif conf.prune.method == "random_weight":
        return "Random Weight"
    elif conf.prune.method == "l1_global":
        return "L¹ Filter Globally"
    return "Not Listed!"

def loss_name(conf):
    try:
        loss_name = conf.loss._target_.split(".")[-1]
    except:
        loss_name=None
    
    # Rename old loss.
    if loss_name == "AugmixKnowledgeDistill":
        return "KnowledgeDistill"
    if loss_name == "AugmixLoss":
        return "StandardLoss"
    return loss_name

def add_hydra(conf, df):
    """ Check whether this run was a fine-tuning run for hydra."""
    path = conf.network.ckpt_path    
    #print('entering add hydra')
    if path is None:
        #print('exiting because path does not exist')
        return
   
    config_path = Path(path).parent.parent / "configs.yaml"
    config_path=Path(str(config_path).replace("hoffmaja", "agnihotr-shashank-pruneshift/hoffmaja"))

    if not config_path.exists():        
        #print(config_path)
        return

    conf_ckpt = OmegaConf.load(config_path)

    if not "subnet" in conf_ckpt:
        #print('exiting because subnet does not exist')
        return
    
    df["Prune Ratio"] = conf_ckpt.subnet.ratio
    # df["Hydra DeepAugment"] = bool(conf_ckpt.datamodule.augmix)
    # df["Hydra Augmix"] = bool(conf_ckpt.datamodule.deepaugment_path)
    df["Hydra Amda"] = bool(conf_ckpt.datamodule.augmix) and bool(conf_ckpt.datamodule.deepaugment_path)
    df["Hydra Loss"] = loss_name(conf_ckpt)
    #print('Adding everything')

def add_supcon(conf, df):
    """ Check whether this run was a fine-tuning run for supcon."""
    path = conf.network.ckpt_path
    if path is None:
        return
   
    config_path = Path(path).parent.parent / "configs.yaml"

    if not config_path.exists():
        return

    conf_ckpt = OmegaConf.load(config_path)

    #if not "SupCon" in conf_ckpt:
        #return
    
    #df["Teacher"] = conf_ckpt.teacher
    #print(conf_ckpt)
    #df["Supcon with KD"] = bool(conf_ckpt.teacher)
    #df["Supcon Augmix"] = bool(conf_ckpt.datamodule.augmix)
    df["Supcon Amda"] = bool(conf_ckpt.datamodule.augmix) and bool(conf_ckpt.datamodule.deepaugment_path)
    df["SupCon Loss"] = loss_name(conf_ckpt)
    df["Temperature"] = conf_ckpt.loss.base_temperature
    df["Batch Size"] = conf_ckpt.datamodule.batch_size

def build_df(path):
    path=Path(str(path))#.replace("/work/dlclarge1/", "/misc/lmbraid19/agnihotr/"))
    if not Path.exists(path / "metrics.csv"):
        return None

    df = pd.read_csv(path / "metrics.csv").iloc[-1: ].filter(regex="test_*")
    conf = OmegaConf.load(path / "configs.yaml")

    if not OmegaConf.is_none(conf, "prune"):
        value = OmegaConf.is_none(conf, "prune")
        print(value)
        df["Prune Method"] = prune_method(conf)
        amount = conf.prune.amount
        ratio = conf.prune.ratio
        if not amount is None and ratio is None:
            if amount is None:
                amount = 1 - 1 / ratio
            if ratio is None:
                ratio = 1 / (1 - amount)
        
        df["Prune Amount"] = amount
        df["Prune Ratio"] = ratio
        
    df["Epochs"] = conf.trainer.max_epochs
    df["Loss"] = loss_name(conf)
    df["Network"] = conf.network.name
    #df["LearningRate"] = conf.optimizer.lr
    #df["WeightDecay"] = conf.optimizer.weight_decay
    #df["kd_T"] = conf.loss.kd_T
    #df["kd_mixture"] = conf.loss.kd_mixture
    
    if not OmegaConf.is_none(conf.network, "ensemble"):
        df["ensemble"] = conf.network.ensemble
        if not OmegaConf.is_none(conf.network, "loading_ensemble"):
            df["loading_ensemble"] = conf.network.loading_ensemble
        else:
            df["loading_ensemble"] = False
        try :
            df["network1"] = conf.network.network1_path
        except:
            df["network1"] = None
        try :
            df["network2"] = conf.network.network2_path
        except:
            df["network2"] = None
        try :
            df["network3"] = conf.network.network3_path
        except:
            df["network3"] = None
        try:
            parent_path = Path(conf.network.network1_path).parent.parent / "configs.yaml"
            
        except:
            print("no parent for this ensemble.")
        else:
            #parent_path=Path(str(parent_path).replace('/work', '/misc'))
            #parent_path=Path(str(parent_path).replace('dlclarge1', 'lmbraid19/agnihotr'))            
            parent_conf = OmegaConf.load(parent_path)            
            if not OmegaConf.is_none(parent_conf, "prune"):
                df["Prune Method"] = prune_method(parent_conf)
                amount = parent_conf.prune.amount
                ratio = parent_conf.prune.ratio
                if not amount is None and ratio is None:
                    if amount is None:
                        amount = 1 - 1 / ratio
                    if ratio is None:
                        ratio = 1 / (1 - amount)
        
                df["Prune Amount"] = amount
                df["Prune Ratio"] = ratio
    else:
        df["ensemble"] = False
        df["loading_ensemble"] = False
        df["network1"] = None
        df["network2"] = None
        df["network3"] = None
    if not OmegaConf.is_none(conf.network, "multiheaded"):
        df["multiheaded"]=True
    else:
        df["multiheaded"]=False
    

        
    
    if not OmegaConf.is_none(conf, "teacher"):
        df["Teacher"] = teacher_name(conf)
        
    if not OmegaConf.is_none(conf.network, "scaling_factor"):
        df["Scaling"] = conf.network.scaling_factor
    else:
        df["Scaling"] = 1

    add_hydra(conf, df)
    add_supcon(conf, df)
        
    #df["Augmix"] = bool(conf.datamodule.augmix)
    #df["DeepAugment"] = bool(conf.datamodule.deepaugment_path)
    #df["Amda"] = df["Augmix"] & df["DeepAugment"]
    df["Augmix"] = True if 'amda' in str(path) else False
    df["DeepAugment"] = True if 'amda' in str(path) else False
    df["Amda"] = df["Augmix"] & df["DeepAugment"]
    

    if "test_acc_clean" in df.columns:
        df[IMG_TAG] = 1 - df["test_acc_clean"]

    if "test_mCE" in df.columns:
        df[IMGC_TAG] = df["test_mCE"]

    if "test_acc_rendition" in df.columns:
        df[IMGR_TAG] = 1 - df["test_acc_rendition"]

    df["Path"] = str(path)
    #df["Path"] = str(path).replace("/work/dlclarge1/", "/misc/lmbraid19/agnihotr/")
        
    non_test_columns = [c for c in df.columns if c[:4] != "test"]
    return df[non_test_columns]

def collect(*paths, err_to_percentage=True):
    exp_folders = set(find_folders(*paths))
    df = pd.concat(filter(lambda d: d is not None, [build_df(p) for p in exp_folders]), sort=True)
    
    if err_to_percentage:
        to_percentage(df)
    return df
