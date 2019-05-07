import numpy as np
from scipy import stats

regression = stats.linregress([0.4, 0.5, 0.6, 0.7, 0.8], [0.1, 0.25, 0.55,0.75, 0.85])
print(regression)
