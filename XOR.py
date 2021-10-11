# IMPORT LIBRARIES
import pandas as pd

# IMPORTING PACKAGES AND MODULES
from utilities.Perceptron_class import Perceptron
from utilities.all_ImpFunctions import prepare_data, save_model, save_plot

# DATASET
XOR = {
    'x1' : [0, 0, 1, 1],
    'x2' : [0, 1, 0, 1],
    'y' : [0, 1, 1, 0]
}
df2 = pd.DataFrame(XOR)
print(df2)

# PREPARE DATA
x, y = prepare_data(df2)

# FIT MODEL
ETA = 0.3 # SHOULD BE BETWEEN 0 AND 1
EPOCHS = 10

model_XOR = Perceptron(ETA, EPOCHS)
model_XOR.fit(x, y)

# PREDICTIONS AND TOTAL LOSS
model_XOR.predict(x)
model_XOR.total_loss()

# SAVE MODEL
save_model(model_XOR, 'XOR.model')

# SAVE PLOT
save_plot(df2, "XOR.png", model_XOR)