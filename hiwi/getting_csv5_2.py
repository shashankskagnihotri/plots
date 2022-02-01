

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




