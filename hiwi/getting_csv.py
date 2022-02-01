

###TODO###
# 1. prune seperate    .......................done
# 2. calibrated pruned seperate ..............done
# 3. prune joint train seperate   ............done
# 4. calibrated prune joint train seperate ....done
# 5. prune and train joint     ................done
# 6. calibrated prune and train joint    .....missing
# 7. Scaled up resnet18     ..................
# 8. Calibrated scaled up resnet18   .........
# 9. Multiheaded  ............................

# 1. Compare errors, test error, ECE, mCE, Rendition
# 2. What layers are being pruned
# 3. What is the network predicting 

from utils import collect, add_info
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np




path=[Path("/work/dlclarge1/agnihotr-ensemble/scaled_resnet18/imagenet100/standard_1.73"), 
     Path("/work/dlclarge1/agnihotr-ensemble/scaled_resnet18/imagenet100/pruned/l1_global/standard")]
scaled_up_std=collect(*path)
scaled_up_std.to_csv('csv/debug_scaled_up_std.csv')
add_info(scaled_up_std)
scaled_up_std = scaled_up_std.sort_values(by="Model Size")
scaled_up_std["Name"]=["Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up"]
scaled_up_std.to_csv('csv/scaled_up_std.csv')
print('scaled up std done')




path=[Path("/work/dlclarge1/agnihotr-ensemble/scaled_resnet18/imagenet100/amda_1.73"), 
     Path("/work/dlclarge1/agnihotr-ensemble/scaled_resnet18/imagenet100/pruned/l1_global/amda")]
scaled_up_amda=collect(*path)
add_info(scaled_up_amda)
scaled_up_amda = scaled_up_amda.sort_values(by="Model Size")
scaled_up_amda["Name"]=["Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up", "Not Calibrated Scaled Up"]
scaled_up_amda.to_csv('csv/scaled_up_amda.csv')
print('scaled up amda done')




path=Path("/work/dlclarge1/agnihotr-ensemble/calibrated_scaled_up/resnet18/amda")
calibrated_scaled_up_amda=collect(path)
add_info(calibrated_scaled_up_amda)
calibrated_scaled_up_amda = calibrated_scaled_up_amda.sort_values(by="Model Size")
calibrated_scaled_up_amda["Name"]=["Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up"]
calibrated_scaled_up_amda.to_csv('csv/calibrated_scaled_up_amda.csv')
print('calibrated scaled up amda done')


path=Path("/work/dlclarge1/agnihotr-ensemble/calibrated_scaled_up/resnet18/standard")
calibrated_scaled_up_std=collect(path)
add_info(calibrated_scaled_up_std)
calibrated_scaled_up_std = calibrated_scaled_up_std.sort_values(by="Model Size")
calibrated_scaled_up_std["Name"]=["Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up"]
calibrated_scaled_up_std.to_csv('csv/calibrated_scaled_up_std.csv')
print('scaled up std done')






seperate_amda_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/second_ensembles/resnet18/amda/performance"))
add_info(seperate_amda_base)
seperate_amda_base = seperate_amda_base.sort_values(by="Model Size")
seperate_amda_base.to_csv('csv/seperate_amda_base.csv')




seperate_std_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/second_ensembles/resnet18/standard/performance"))
add_info(seperate_std_base)
seperate_std_base = seperate_std_base.sort_values(by="Model Size")
seperate_std_base.to_csv('csv/seperate_std_base.csv')
print('seperated std base done')




seperate_amda_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/second_ensembles/resnet18/amda/prune_sep/performance"))
add_info(seperate_amda_pruned)
seperate_amda_pruned = seperate_amda_pruned.sort_values(by="Model Size")
seperate_amda_pruned.to_csv('csv/seperate_amda_pruned.csv')
print('seperated amda pruned done')




seperate_std_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/second_ensembles/resnet18/standard/prune_sep/performance"))
add_info(seperate_std_pruned)
seperate_std_pruned = seperate_std_pruned.sort_values(by="Model Size")
seperate_std_pruned.to_csv('csv/seperate_std_pruned.csv')
print('seperated standard pruned done')



seperate_amda_base=pd.read_csv('csv/seperate_amda_base.csv')
seperate_amda_base["Prune Amount"]=0
seperate_amda_pruned=pd.read_csv('csvs/seperate_amda_pruned.csv')
seperate_amda=seperate_amda_base.append(seperate_amda_pruned, ignore_index=True)
seperate_amda=seperate_amda.sort_values(by="Model Size")
seperate_amda["Name"] = ["Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate"]
seperate_amda.to_csv('csv/not_calibrated_seperate_amda.csv')
print('not_calibrated_seperate_amda done')



seperate_std_base=pd.read_csv('csv/seperate_std_base.csv')
seperate_std_pruned=pd.read_csv('csv/seperate_std_pruned.csv')
seperate_std=seperate_std_base.append(seperate_std_pruned, ignore_index=True)
seperate_std=seperate_std.sort_values(by="Model Size")
seperate_std["Name"] = ["Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate"]
seperate_std.to_csv('csv/not_calibrated_seperate_std.csv')
print('not_calibrated_seperate_std done')







path=[Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/standard/finetuned/10/performance"), 
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/standard/finetuned/25/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/standard/finetuned/50/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/standard/finetuned/70/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/standard/finetuned/90/performance")]
pruned_jointly_std_pruned=collect(*path)
add_info(pruned_jointly_std_pruned)
pruned_jointly_std_pruned = pruned_jointly_std_pruned.sort_values(by="Model Size")
pruned_jointly_std_pruned.to_csv('csv/not_calibrated_pruned_jointly_std_pruned.csv')
print('not_calibrated_pruned jointly std pruned done')



path=[Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/amda/finetuned/10/performance"), 
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/amda/finetuned/25/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/amda/finetuned/50/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/amda/finetuned/70/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/not_calibrated/amda/finetuned/90/performance")]
pruned_jointly_amda_pruned=collect(*path)
add_info(pruned_jointly_amda_pruned)
pruned_jointly_amda_pruned = pruned_jointly_amda_pruned.sort_values(by="Model Size")
pruned_jointly_amda_pruned.to_csv('csv/not_calibrated_pruned_jointly_amda_pruned.csv')
print('not_calibrated_pruned jointly amda pruned done')



seperate_std_base=pd.read_csv('csv/seperate_std_base.csv')
pruned_jointly_std_pruned=pd.read_csv('csv/not_calibrated_pruned_jointly_std_pruned.csv')
pruned_jointly_std=seperate_std_base.append(pruned_jointly_std_pruned, ignore_index=True)
pruned_jointly_std=pruned_jointly_std.sort_values(by="Model Size")
pruned_jointly_std["Name"] = ["Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate"]
pruned_jointly_std.to_csv('csv/not_calibrated_pruned_jointly_std.csv')
print('not_calibrated_pruned jointly std done')





seperate_amda_base=pd.read_csv('csv/seperate_amda_base.csv')
pruned_jointly_amda_pruned=pd.read_csv('csv/not_calibrated_pruned_jointly_amda_pruned.csv')
pruned_jointly_amda=seperate_amda_base.append(pruned_jointly_amda_pruned, ignore_index=True)
pruned_jointly_amda=pruned_jointly_amda.sort_values(by="Model Size")
pruned_jointly_amda["Name"] = ["Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate","Not Calibrated Prune joint Train seperate"]
pruned_jointly_amda.to_csv('csv/not_calibrated_pruned_jointly_amda.csv')
print('not_calibrated_pruned jointly amda done')









path=[Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/standard/finetuned/10/performance"), 
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/standard/finetuned/25/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/standard/finetuned/50/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/standard/finetuned/70/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/standard/finetuned/90/performance")]
calibrated_pruned_jointly_std_pruned=collect(*path)
add_info(calibrated_pruned_jointly_std_pruned)
calibrated_pruned_jointly_std_pruned = calibrated_pruned_jointly_std_pruned.sort_values(by="Model Size")
calibrated_pruned_jointly_std_pruned.to_csv('csv/calibrated_pruned_jointly_std_pruned.csv')
print('calibrated_pruned jointly std pruned done')



path=[Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/amda/finetuned/10/performance"), 
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/amda/finetuned/25/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/amda/finetuned/50/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/amda/finetuned/70/performance"),
     Path("/work/dlclarge1/agnihotr-ensemble/second_prune_joint_train_sep/calibrated/amda/finetuned/90/performance")]
calibrated_pruned_jointly_amda_pruned=collect(*path)
add_info(calibrated_pruned_jointly_amda_pruned)
calibrated_pruned_jointly_amda_pruned = calibrated_pruned_jointly_amda_pruned.sort_values(by="Model Size")
calibrated_pruned_jointly_amda_pruned.to_csv('csv/calibrated_pruned_jointly_amda_pruned.csv')
print('calibrated_pruned jointly amda pruned done')



calibrated_seperate_std_base=pd.read_csv('csv/calibrated_seperate_std_base.csv')
calibrated_pruned_jointly_std_pruned=pd.read_csv('csv/calibrated_pruned_jointly_std_pruned.csv')
calibrated_pruned_jointly_std=calibrated_seperate_std_base.append(calibrated_pruned_jointly_std_pruned, ignore_index=True)
calibrated_pruned_jointly_std=calibrated_pruned_jointly_std.sort_values(by="Model Size")
calibrated_pruned_jointly_std["Name"] = ["Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate"]
calibrated_pruned_jointly_std.to_csv('csv/calibrated_pruned_jointly_std.csv')
print('calibrated_pruned jointly std done')



calibrated_seperate_amda_base=pd.read_csv('csv/calibrated_seperate_amda_base.csv')
calibrated_pruned_jointly_amda_pruned=pd.read_csv('csv/calibrated_pruned_jointly_amda_pruned.csv')
calibrated_pruned_jointly_amda=calibrated_seperate_amda_base.append(calibrated_pruned_jointly_amda_pruned, ignore_index=True)
calibrated_pruned_jointly_amda=calibrated_pruned_jointly_amda.sort_values(by="Model Size")
calibrated_pruned_jointly_amda["Name"] = ["Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate"]
calibrated_pruned_jointly_amda.to_csv('csv/calibrated_pruned_jointly_amda.csv')
print('calibrated_pruned jointly amda done')











together_amda_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/together_second/resnet18/amda"))
add_info(together_amda_base)
together_amda_base = together_amda_base.sort_values(by="Model Size")
together_amda_base.to_csv('csv/together_amda_base.csv')
print('together amda base done')



together_std_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/together_second/resnet18/standard"))
add_info(together_std_base)
together_std_base = together_std_base.sort_values(by="Model Size")
together_std_base.to_csv('csv/together_std_base.csv')
print('together std base done')





together_amda_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/together_second/resnet18/pruned_amda"))
add_info(together_amda_pruned)
together_amda_pruned = together_amda_pruned.sort_values(by="Model Size")
together_amda_pruned.to_csv('csv/together_amda_pruned.csv')
print('together amda pruned done')



together_std_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/together_second/resnet18/pruned_standard"))
add_info(together_std_pruned)
together_std_pruned = together_std_pruned.sort_values(by="Model Size")
together_std_pruned.to_csv('csv/together_std_pruned.csv')




together_std_base=pd.read_csv('csv/together_std_base.csv')
together_std_pruned=pd.read_csv('csv/together_std_pruned.csv')
together_std=together_std_base.append(together_std_pruned, ignore_index=True)
together_std=together_std.sort_values(by="Model Size")
together_std["Name"] = ["Prune-Train jointly","Prune-Train jointly","Prune-Train jointly","Prune-Train jointly","Prune-Train jointly","Prune-Train jointly"]
together_std.to_csv('csv/together_std.csv')
print('together std done')



together_amda_base=pd.read_csv('csv/together_amda_base.csv')
together_amda_pruned=pd.read_csv('csv/together_amda_pruned.csv')
together_amda=together_amda_base.append(together_amda_pruned, ignore_index=True)
together_amda=together_amda.sort_values(by="Model Size")
together_amda["Name"] = ["Prune-Train jointly","Prune-Train jointly","Prune-Train jointly","Prune-Train jointly","Prune-Train jointly","Prune-Train jointly"]
together_amda.to_csv('csv/together_amda.csv')
print('together amda done')





calibrated_seperate_amda_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/calibrated/resnet18/amda/unpruned/performance"))
add_info(calibrated_seperate_amda_base)
calibrated_seperate_amda_base = calibrated_seperate_amda_base.sort_values(by="Model Size")
calibrated_seperate_amda_base.to_csv('csv/calibrated_seperate_amda_base.csv')
print('calibrated seperate amda base done')



calibrated_seperate_std_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/calibrated/resnet18/standard/unpruned/performance"))
add_info(calibrated_seperate_std_base)
calibrated_seperate_std_base = calibrated_seperate_std_base.sort_values(by="Model Size")
calibrated_seperate_std_base.to_csv('csv/calibrated_seperate_std_base.csv')
print('calibrated seperate std base done')



calibrated_seperate_amda_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/calibrated/resnet18/amda/pruned/performance"))
add_info(calibrated_seperate_amda_pruned)
calibrated_seperate_amda_pruned = calibrated_seperate_amda_pruned.sort_values(by="Model Size")
calibrated_seperate_amda_pruned.to_csv('csv/calibrated_seperate_amda_pruned.csv')
print('calibrated seperate amda pruned done')



calibrated_seperate_std_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/calibrated/resnet18/standard/pruned/performance"))
add_info(calibrated_seperate_std_pruned)
calibrated_seperate_std_pruned = calibrated_seperate_std_pruned.sort_values(by="Model Size")
calibrated_seperate_std_pruned.to_csv('csv/calibrated_seperate_std_pruned.csv')
print('calibrated seperate std pruned done')



calibrated_seperate_amda_base=pd.read_csv('csv/calibrated_seperate_amda_base.csv')
calibrated_seperate_amda_pruned=pd.read_csv('csv/calibrated_seperate_amda_pruned.csv')
calibrated_seperate_amda=calibrated_seperate_amda_base.append(calibrated_seperate_amda_pruned, ignore_index=True)
calibrated_seperate_amda=calibrated_seperate_amda.sort_values(by="Model Size")
calibrated_seperate_amda["Name"] = ["Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate"]
calibrated_seperate_amda.to_csv('csv/calibrated_seperate_amda.csv')
print('calibrated seperate amda done')



calibrated_seperate_std_base=pd.read_csv('csv/calibrated_seperate_std_base.csv')
calibrated_seperate_std_pruned=pd.read_csv('csv/calibrated_seperate_std_pruned.csv')
calibrated_seperate_std=calibrated_seperate_std_base.append(calibrated_seperate_std_pruned, ignore_index=True)
calibrated_seperate_std=calibrated_seperate_std.sort_values(by="Model Size")
calibrated_seperate_std["Name"] = ["Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate","Calibrated Prune-Train seperate"]
calibrated_seperate_std.to_csv('csv/calibrated_seperate_std.csv')
print('calibrated seperate std done')









baseline_path = Path("/work/dlclarge1/agnihotr-ensemble/baselines")
baseline=collect(baseline_path).query("Scaling == 1")

std_baseline=baseline.query("Network=='resnet18' and not Amda")
add_info(std_baseline)
std_baseline.to_csv('csv/baseline_std.csv')
print('baseline std done')



baseline_path = Path("/work/dlclarge1/agnihotr-ensemble/baselines")
baseline=collect(baseline_path).query("Scaling == 1")

amda_baseline=baseline.query("Network=='resnet18' and Amda")
add_info(amda_baseline)
amda_baseline.to_csv('csv/baseline_amda.csv')
print('baseline amda done')





multi_std_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100")).query("Path=='/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100/standard'")
multi_std_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100/standard/l1_global"))
multi_std=multi_std_base.append(multi_std_pruned)
add_info(multi_std)
multi_std=multi_std.sort_values(by="Model Size")
multi_std["Name"]=["Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble"]
multi_std.to_csv('csv/mutil_std.csv')
print('multiheaded std done')



multi_amda_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100")).query("Path=='/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100/amda'")
multi_amda_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100/amda/l1_global_actual"))
multi_amda=multi_amda_base.append(multi_amda_pruned)
add_info(multi_amda)
multi_amda=multi_amda.sort_values(by="Model Size")
multi_amda["Name"]=["Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble"]
multi_amda.to_csv('csv/mutil_amda.csv')
print('multiheaded amda done')






