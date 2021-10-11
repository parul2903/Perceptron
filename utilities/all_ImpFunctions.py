# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib
import os

##### (1) PREPARE DATA FUNCTION

def prepare_data(df):
  x = df.drop('y', axis = 1)
  y = df['y']

  return x, y

##### (2) SAVE MODEL FUNCTION

# SAVING OUR MODEL USING JOBLIB LIBRARY
# WE CAN ALSO SAVE OUR MODEL USING PICKLE LIBRARY

def save_model(model, filename):
  model_dir = 'Models'
  os.makedirs(model_dir, exist_ok = True) 
# We use exist_ok because if a directory with the same name is already present, then nothing will happen
# A directory will only create when no directory of same name exists

  filePath = os.path.join(model_dir, filename) # Models/filename
# This os.path.join will create a path for our final according to our system

  joblib.dump(model, filePath)

##### (3) SAVE PLOT FUNCTION

def save_plot(df, file_name, model):
# we have used underscore while creating this function because we need to make sure this is protected 
# and can only be used inside the function save_plot, although we can access it outside as well but we will not
  def _create_base_plot(df): # base plot of the model
    df.plot(kind = 'scatter', x = 'x1', y = 'x2', c = 'y', s = 100, cmap = 'winter')
    plt.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 1) # because y = 0 on the x-axis
    plt.axvline(x = 0, color = 'black', linestyle = '--', linewidth = 1) # because x = 0 on the y-axis
    figure = plt.gcf() # GET CURRENT FIGURE
    figure.set_size_inches(10, 8) # set the size of the figure in inches

  def _plot_decision_regions(x, y, classfier, resolution=0.02): # plotting decision boundaries in order to show the classification
    colors = ['red', 'blue', 'lightgreen', 'cyan', 'grey']
    cmap = ListedColormap(colors[:len(np.unique(y))]) # y is the classes that is 0 and 1 # len(np.unique(y)) = 2
    # so, cmap will take first two colors that is red and blue

    x = x.values
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1 

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()

  x, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(x, y, model)

  plot_dir = "Plots"
  os.makedirs(plot_dir, exist_ok = True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)