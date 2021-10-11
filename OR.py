# IMPORT LIBRARIES
import pandas as pd

# IMPORTING PACKAGES AND MODULES
from utilities.perceptron_model import Perceptron
from utilities.all_IMPfunctions import prepare_data, save_model, save_plot

# DATASET
OR = {
    'x1' : [0, 0, 1, 1],
    'x2' : [0, 1, 0, 1],
    'y' : [0, 1, 1, 1]
}
df1 = pd.DataFrame(OR)
print(df1)

# PREPARE DATA
x, y = prepare_data(df1)

# FIT MODEL
ETA = 0.3 # SHOULD BE BETWEEN 0 AND 1
EPOCHS = 10

model_OR = Perceptron(ETA, EPOCHS)
model_OR.fit(x, y)

# PREDICTIONS AND TOTAL LOSS
model_OR.predict(x)
model_OR.total_loss()

# SAVE PLOT
save_plot(df1, "AND.png", model_OR)