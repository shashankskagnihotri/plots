

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



multi_amda_base=collect(Path("/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100")).query("Path=='/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100/amda'")
multi_amda_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/multiheaded/imagenet100/amda/l1_global_actual"))
multi_amda=multi_amda_base.append(multi_amda_pruned)
add_info(multi_amda)
multi_amda=multi_amda.sort_values(by="Model Size")
multi_amda["Name"]=["Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble", "Multiheaded Ensemble"]
multi_amda.to_csv('csv/mutil_amda.csv')
print('multiheaded amda done')






