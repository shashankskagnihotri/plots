

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








together_amda_pruned=collect(Path("/work/dlclarge1/agnihotr-ensemble/together_second/resnet18/pruned_amda"))
add_info(together_amda_pruned)
together_amda_pruned = together_amda_pruned.sort_values(by="Model Size")
together_amda_pruned.to_csv('csv/together_amda_pruned.csv')
print('together amda pruned done')


