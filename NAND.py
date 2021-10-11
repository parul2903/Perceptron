# IMPORT LIBRARIES
import pandas as pd

# IMPORTING PACKAGES AND MODULES
from utilities.perceptron_model import Perceptron
from utilities.all_IMPfunctions import prepare_data, save_model, save_plot

# DATASET
NAND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [1,1,1,0],
}
df3 = pd.DataFrame(NAND)
print(df3)

# PREPARE DATA
x, y = prepare_data(df3)

# FIT MODEL
ETA = 0.3 # SHOULD BE BETWEEN 0 AND 1
EPOCHS = 10

model_NAND = Perceptron(ETA, EPOCHS)
model_NAND.fit(x, y)

# PREDICTIONS AND TOTAL LOSS
model_NAND.predict(x)
model_NAND.total_loss()

# SAVE PLOT
save_plot(df3, "NAND.png", model_NAND)

