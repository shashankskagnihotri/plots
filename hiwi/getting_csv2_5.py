

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







seperate_std_base=pd.read_csv('csv/seperate_std_base.csv')
seperate_std_pruned=pd.read_csv('csv/seperate_std_pruned.csv')
seperate_std=seperate_std_base.append(seperate_std_pruned, ignore_index=True)
seperate_std=seperate_std.sort_values(by="Model Size")
seperate_std["Name"] = ["Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate","Not Calibrated Prune-Train seperate"]
seperate_std.to_csv('csv/not_calibrated_seperate_std.csv')
print('not_calibrated_seperate_std done')




