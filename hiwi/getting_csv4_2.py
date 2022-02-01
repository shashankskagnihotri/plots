

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




calibrated_seperate_amda_base=pd.read_csv('csv/calibrated_seperate_amda_base.csv')
calibrated_pruned_jointly_amda_pruned=pd.read_csv('csv/calibrated_pruned_jointly_amda_pruned.csv')
calibrated_pruned_jointly_amda=calibrated_seperate_amda_base.append(calibrated_pruned_jointly_amda_pruned, ignore_index=True)
calibrated_pruned_jointly_amda=calibrated_pruned_jointly_amda.sort_values(by="Model Size")
calibrated_pruned_jointly_amda["Name"] = ["Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate","Calibrated Prune joint Train seperate"]
calibrated_pruned_jointly_amda.to_csv('csv/calibrated_pruned_jointly_amda.csv')
print('calibrated_pruned jointly amda done')



