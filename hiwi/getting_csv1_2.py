

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





path=Path("/work/dlclarge1/agnihotr-ensemble/calibrated_scaled_up/resnet18/standard")
calibrated_scaled_up_std=collect(path)
add_info(calibrated_scaled_up_std)
calibrated_scaled_up_std = calibrated_scaled_up_std.sort_values(by="Model Size")
calibrated_scaled_up_std["Name"]=["Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up", "Calibrated Scaled Up"]
calibrated_scaled_up_std.to_csv('csv/calibrated_scaled_up_std.csv')
print('calibrated scaled up std done')




