# IMPORT LIBRARIES
import pandas as pd

# IMPORTING PACKAGES AND MODULES
#from utilities.Perceptron_class import Perceptron
from oneNeuron.perceptron import Perceptron
from utilities.all_ImpFunctions import prepare_data, save_model, save_plot
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok = True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"), filemode = 'a', level = logging.INFO, format = logging_str)
# filemode = 'a': all the logs will get on appending in the file # default filemode is 'a' 
# logs/running_logs.log
# We are creating this file so that all those things which are printed in the terminal can get printed the
# newly created file instead.


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
    OR = {
        'x1' : [0, 0, 1, 1],
        'x2' : [0, 1, 0, 1],
        'y' : [0, 1, 1, 1]
    }

    ETA = 0.3 # SHOULD BE BETWEEN 0 AND 1
    EPOCHS = 10

    try:
        logging.info(">>> starting training >>>")
        main(data = OR, eta = ETA, epochs = EPOCHS, modelFilename = 'OR.model', plotFilename = 'OR.png')
        logging.info("<<< training finished successfully <<<")
    except Exception as e:
        logging.exception(e)
        raise e # this will raise error in the terminal as well, if any # earlier the error was only shown in the log file

    # when we use logging.exception, it only logs the error message instead of entire exception