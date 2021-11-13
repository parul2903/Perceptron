# We can mention developer details to every file as
"""
author: Parul Sharma
email: parul3kin@gmail.com
"""

# IMPORT LIBRARIES
import pandas as pd

# IMPORTING PACKAGES AND MODULES
from utilities.Perceptron_class import Perceptron
from utilities.all_ImpFunctions import prepare_data, save_model, save_plot
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level = logging.INFO, format = logging_str)

# DATASET

def main(data, eta, epochs, modelFilename, plotFilename):
    
    df = pd.DataFrame(data)
    logging.info(f"This is the actual Data Frame \n{df}")

    # PREPARE DATA
    x, y = prepare_data(df)

    # FIT MODEL
    model = Perceptron(ETA, EPOCHS)
    model.fit(x, y)

    # PREDICTIONS AND TOTAL LOSS
    model.predict(x)
    model.total_loss()

    # SAVE MODEL
    save_model(model, modelFilename)

    # SAVE PLOT
    save_plot(df, plotFilename, model)

if __name__ == '__main__': # entry point from where the code will start executing
    XOR = {
    'x1' : [0, 0, 1, 1],
    'x2' : [0, 1, 0, 1],
    'y' : [0, 1, 1, 0]
}

    ETA = 0.3 # SHOULD BE BETWEEN 0 AND 1
    EPOCHS = 10

    try:
        logging.info(">>> starting training >>>")
        main(data = XOR, eta = ETA, epochs = EPOCHS, modelFilename = 'XOR.model', plotFilename = 'XOR.png')
        logging.info("<<< training finished successfully <<<")
    except Exception as e:
        logging.exception(e)