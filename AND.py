# IMPORT LIBRARIES
import pandas as pd

# IMPORTING PACKAGES AND MODULES
from utilities.Perceptron_class import Perceptron
from utilities.all_ImpFunctions import prepare_data, save_model, save_plot

# DATASET
AND = {
    'x1' : [0, 0, 1, 1],
    'x2' : [0, 1, 0, 1],
    'y' : [0, 0, 0, 1]
}
df = pd.DataFrame(AND)
print(df)

# PREPARE DATA
x, y = prepare_data(df)

# FIT MODEL
ETA = 0.3 # SHOULD BE BETWEEN 0 AND 1
EPOCHS = 10

model_AND = Perceptron(ETA, EPOCHS)
model_AND.fit(x, y)

# PREDICTIONS AND TOTAL LOSS
model_AND.predict(x)
model_AND.total_loss()

# SAVE MODEL
save_model(model_AND, 'AND.model')

# SAVE PLOT
save_plot(df, "AND.png", model_AND)