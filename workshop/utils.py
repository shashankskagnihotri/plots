from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd


IMGR_TAG = "ImageNet100-R Error"
IMGC_TAG = "ImageNet100-C Error"
IMG_TAG = "ImageNet100 Error"


def find_folders(*paths):
    exp_paths = []
    for path in paths:
        exp_paths.extend(Path(path).glob("**/metrics.csv"))
        
    return [p.parent for p in exp_paths]

def teacher_name(conf):
    if conf.teacher.name == "swsl_resnet50":
        return conf.teacher.name
    elif conf.teacher.name == "resnet50":
        path = conf.teacher.ckpt_path
        if path is None:
            return "std_resnet50"
        elif "augmix" in path and "deepaugment" in path:
            return "amda_resnet50"
        elif "augmix" in path:
            return "augmix_resnet50"
        elif "deepaugment" in path:
            return "deep_resnet50"
    return "custom"
    raise NotImplementedError


def loss_name(conf):
    loss_name = conf.loss._target_.split(".")[-1]
    # Rename old loss.
    if loss_name == "AugmixKnowledgeDistill":
        return "KnowledgeDistill"
    if loss_name == "AugmixLoss":
        return "StandardLoss"
    return loss_name


def build_df(path):
    if not Path.exists(path / "metrics.csv"):
        return None

    df = pd.read_csv(path / "metrics.csv").iloc[-1: ].filter(regex="test_*")
    conf = OmegaConf.load(path / "configs.yaml")

    if conf.prune is not None:
        method = conf.prune.method
        if method == "unwider_resnet":
            method = "uniform"
        df["Prune Method"] = method
        df["Prune Amount"] = conf.prune.amount

    df["Loss"] = loss_name(conf)
    df["Network"] = conf.network.name

    if conf.teacher is not None:
        df["Teacher"] = teacher_name(conf)
        
    if conf.network.scaling_factor is not None:
        df["Scaling"] = conf.network.scaling_factor
    else:
        df["Scaling"] = 1

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


def collect(*paths):
    exp_folders = find_folders(*paths)
    df = pd.concat(filter(lambda d: d is not None, [build_df(p) for p in exp_folders]), sort=True)
    return df