from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from multiprocessing import Pool

from pruneshift.networks import create_network
from pruneshift.prune_info import PruneInfo
from pruneshift.utils import get_model_complexity_prune

IMGR_TAG = "ImageNet100-R Error"
IMGC_TAG = "ImageNet100-C Error"
IMG_TAG = "ImageNet100 Error"


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
    if not path.exists():
        print(f"Did not find {path}.")
        return -1, -1
    net = create_network("imagenet", row["Network"], 100, 
                         ckpt_path=path, 
                         scaling_factor=row["Scaling"])#imagenet
    size = PruneInfo(net).network_size()
    macs, _ = get_model_complexity_prune(net, input_res=(3, 224, 224))
    # print(macs)
    return macs, size / 1e6

def add_info(df, num_workers=8):
    rows_iter = (row for _, row in df[["Path", "Scaling", "Network"]].iterrows())
    
    with Pool(num_workers) as pool:
        df[["MACs", "Model Size"]] = pool.map(calculate_info, rows_iter)

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
    if not Path.exists(path / "metrics.csv"):
        return None

    df = pd.read_csv(path / "metrics.csv").iloc[-1: ].filter(regex="test_*")
    conf = OmegaConf.load(path / "configs.yaml")

    if conf.prune is not None:
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
    df["LearningRate"] = conf.optimizer.lr
    df["WeightDecay"] = conf.optimizer.weight_decay
    df["kd_T"] = conf.loss.kd_T
    df["kd_mixture"] = conf.loss.kd_mixture
    
    if conf.network.ensemble:
        df["ensemble"] = conf.network.ensemble
        df["network1"] = conf.network.network1_path
        df["network2"] = conf.network.network2_path
        df["network3"] = conf.network.network3_path
    

        
    
    if conf.teacher is not None:
        df["Teacher"] = teacher_name(conf)
        
    if conf.network.scaling_factor is not None:
        df["Scaling"] = conf.network.scaling_factor
    else:
        df["Scaling"] = 1

    add_hydra(conf, df)
    add_supcon(conf, df)
        
    df["Augmix"] = bool(conf.datamodule.augmix)
    df["DeepAugment"] = bool(conf.datamodule.deepaugment_path)
    df["Amda"] = df["Augmix"] & df["DeepAugment"]

    if "test_acc_clean" in df.columns:
        df[IMG_TAG] = 1 - df["test_acc_clean"]

    if "test_mCE" in df.columns:
        df[IMGC_TAG] = df["test_mCE"]

    if "test_acc_rendition" in df.columns:
        df[IMGR_TAG] = 1 - df["test_acc_rendition"]

    df["Path"] = str(path)
        
    non_test_columns = [c for c in df.columns if c[:4] != "test"]
    return df[non_test_columns]

def collect(*paths, err_to_percentage=True):
    exp_folders = set(find_folders(*paths))
    df = pd.concat(filter(lambda d: d is not None, [build_df(p) for p in exp_folders]), sort=True)
    
    if err_to_percentage:
        to_percentage(df)
    return df