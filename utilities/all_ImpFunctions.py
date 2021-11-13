# For docstrings, we cn use 2 methods. First is the one that we have shown below
# second method:
"""
It is used to save the plot 
:param df: its a data frame 
:param file_name: its a pth to save the plot
:param model: trained model
"""

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib
import os
import logging

##### (1) PREPARE DATA FUNCTION

def prepare_data(df):
  """It is used to separate the independent variables (features) and dependent variables (labels)

  Args:
      df (pd.DataFrame): It is the Pandas Data Frame

  Returns:
      DataFrame: It returns the Data Frames of the independent and dependent variables
  """
  logging.info("Preparing the model by segregating independent and dependent variables")

  x = df.drop('y', axis = 1)
  y = df['y']

  return x, y

##### (2) SAVE MODEL FUNCTION

# SAVING OUR MODEL USING JOBLIB LIBRARY
# WE CAN ALSO SAVE OUR MODEL USING PICKLE LIBRARY

def save_model(model, filename):
  """It is used to save the  trained model

  Args:
      model (python object): trained model
      filename (str): path to save the model
  """
  logging.info("saving the trained model")

  model_dir = 'Models'
  os.makedirs(model_dir, exist_ok = True) 
# We use exist_ok because if a directory with the same name is already present, then nothing will happen
# A directory will only create when no directory of same name exists

  filePath = os.path.join(model_dir, filename) # Models/filename
# This os.path.join will create a path for our file according to our system

  joblib.dump(model, filePath)

  logging.info(f"saved the trained model at {filePath}")

##### (3) SAVE PLOT FUNCTION

def save_plot(df, file_name, model):
  """It is uses to save the plot

  Args:
      df (DataFrame): It is a Data Frame object
      file_name (str): It is the path to save the plot
      model (python object): trained model
  """
  logging.info(f"saving the plot")
# we have used underscore while creating this function because we need to make sure this is protected 
# and can only be used inside the function save_plot, although we can access it outside as well but we will not
  def _create_base_plot(df): # base plot of the model
    """It is uded to create the base plot that is, axis lines, plotting of numbers 

    Args:
        df (DataFrame): It is a Data Frame
    """
    logging.info("creating the base plot")
    df.plot(kind = 'scatter', x = 'x1', y = 'x2', c = 'y', s = 100, cmap = 'winter') # s: size of data points
    plt.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 1) # because y = 0 on the x-axis
    plt.axvline(x = 0, color = 'black', linestyle = '--', linewidth = 1) # because x = 0 on the y-axis
    figure = plt.gcf() # GET CURRENT FIGURE
    figure.set_size_inches(10, 8) # set the size of the figure in inches

  def _plot_decision_regions(x, y, classfier, resolution=0.02): # plotting decision boundaries in order to show the classification
    """It is used to create the decision boundary depending on ddifferent models

    Args:
        x (DataFrame): It is a Data Frame containing x1 and x2 values
        y (DataFrame): It is a Data Frame containing actual output values (y)
        classfier (Python object): Trained model
        resolution (float, optional): . Defaults to 0.02.
    """
    logging.info("plotting the decision boundary")
    colors = ['red', 'blue', 'lightgreen', 'cyan', 'grey']
    cmap = ListedColormap(colors[:len(np.unique(y))]) 
    # y is the classes that is 0 and 1 
    # len(np.unique(y)) = 2
    # so, cmap will take first two colors that is red and blue

    x = x.values
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1 

    # np.meshgrid(): The numpy module of Python provides meshgrid() function for creating a rectangular grid
    # with the help of the given 1-D arrays that represent the Matrix indexing or Cartesian indexing.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), # resolution: difference between the values of two data points
                           np.arange(x2_min, x2_max, resolution))
    # np.ravel(): The numpy.ravel() functions returns contiguous flattened array
    # (1D array with all the input-array elements and with the same type as it)
    z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    # contour and contourf draw contour lines and filled contours, respectively.
    # alpha: values lies between 0(transparent) and 1(opaque)
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
  logging.info(f"saved the plots at {plotPath}")