#!/usr/bin/env python
# coding: utf-8

# # Dataset Generator notebook for Magnetic Signal reconstruction
# 
# This notebook describes explicitly the magnetic field component dataset generator's first version. Because Matlab requests specifics version of python to load its engine, we are using an old python version. Python 2.7 seems to be more stable with the Matlab version I had (R2016b) regarding long-running time and potential data leaks. The versions can be updated easily. I wrote the code allowing it to run on a Python3 environment needing as fewer modifications as possible.
# 
# I don't recommand to use this Notebook to generate the training set, using a python script is much easier and faster.
# To generate a .py script from a notebook (.ipynb) run this command:
# 
# `jupyter nbconvert --to python notebook_name.ipynb`
# 
# This will generate a python script (with all the comments included) you can run in background using for example:
# 
# `( nohup python script.py & )` 
# 
# `nohup` launches the script as a background job. If you launch this through `ssh` the job will be killed after you close the session. To avoid that, add parenthesis to run it in a subshell. I recommand using `nohup` to keep trace of what's happening through execution in the nohup file. It can be consulted any time of the execution using `tail -f nohup.out` to access in real time the last written bits.

# In[2]:

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=DataConversionWarning) 
import matlab.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import datetime
from scipy.io import loadmat
from collections import defaultdict
from os import system
import math
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from IPython.display import clear_output
import Tkinter as tk
import ttk
from tkFont import Font
import threading
import StringIO


# Python allows us to run Matlab functions in background and retrieve their output. The Output format will, of course, be specific, for example, a `float ` output coming out from a Matlab function will be interpreted as `matlab.double` for Linux and `matlab.mlarray.double` for Windows.
# All of these can be reformatted to native python variables. We now can call Matlab functions from the `eng` object. 
# 
# __*Only functions referenced in the installed Matlab's path can be called from the engine*__

# In[3]:


eng = matlab.engine.start_matlab() #starting and storing the matlab engine


# ## Variables instantiation
# The below cell instantiates all the variables we will use in the execution. Working with this amount of `public` variables is not a good practice. But for development purpose and easier debugging time (Jupyter doesn't have any debugger natively), instantiating them as `public` allows us to access them at any time in the process easily. One improvement would be to install an external Debugging plugin to Jupyter-lab and transpose all the `public` variables to `private`.

# In[4]:


quietDays = np.array(eng.quiet()) # we retreive the output of the `quiet()` matlab function, convert it as `numpy.array` and finally store it.

# Setting up the paths (relative path obviously, needs to be changed if the folder structure changes)
trainingDatasetPath = "../TrainingDataset"
trainingDatasetPathASIA = "../TrainingDataset/Asia/"

# load the stations list
mat = loadmat("../../../../opt/dir.indices/station.mat")

# store the "station" column values from the .mat file 
stationsList = mat.get("station", "none")[0]
stationCodesID = dict()
stationIDlatMag = dict()
allStationCodes=np.array([])
allStationLatgeos=np.array([])
allStationLongeos=np.array([])

# store in separate arrays all the stations code (clf, aae...) and their geographic latitudes
a=1
for x in stationsList:
    stationIDlatMag[x[1][0]] = int(x[10][0][0])
    stationCodesID[x[1][0]] = a
    allStationCodes=np.append(allStationCodes,x[1][0])
    allStationLatgeos=np.append(allStationLatgeos,x[3][0])
    allStationLongeos = np.append(allStationLongeos,x[2][0])
    a+=1


# ### EmptyVariables()
# 
# Free all the arrays and dictionaries memory for a fresh loop restart of the following day.

# In[5]:


def EmptyVariables():
    global stationIndicatorRatioVariation, indicatorVariationArray, indicatorVariationArrayLocalTime, normalized01StationIndicatorVariation, ReconstructedArray, stationsWithNoData, stationsNonexistentInFolder, stationsOut, stationIndicatorVariation
    
    stationIndicatorRatioVariation = defaultdict(list)
    indicatorVariationArray = np.array([])
    indicatorVariationArrayLocalTime = np.array([])
    normalized01StationIndicatorVariation = defaultdict(list)
    ReconstructedArray = np.array([])
    stationsWithNoData = []
    stationsNonexistentInFolder = []
    stationsOut = dict()
    stationIndicatorVariation = defaultdict(list)


# ### ChoosePresetArea(`str`, `list`)
# 
# Outputs a list of max/min longitudes and latitudes according to specific preset, mainly for a gain of time. For now 3 presets are available and can be selected by the name of the area. For example, 'america' will return [30,54,240,307] corresponding to min lat, max lat, min long, max long. There is obviously a way to enter a custom area by selecting 'custom' and writing the degrees of the wanted zone.

# In[6]:


def ChoosePresetArea(area,customArea):
    america = [30,54,240,307]
    asia = [30,54,57,159]
    europe = [40,64,0,50]
    laMin = 0
    laMax = 0 
    lonMin = 0 
    lonMax = 0
    if area == 'america':
        laMin=america[0]
        laMax=america[1]
        lonMin=america[2]
        lonMax=america[3]
    elif area == 'asia':
        laMin=asia[0]
        laMax=asia[1]
        lonMin=asia[2]
        lonMax=asia[3]
    elif area == 'europe':
        laMin=europe[0]
        laMax=europe[1]
        lonMin=europe[2]
        lonMax=europe[3]
    elif area == 'custom':
        laMin=customArea[0]
        laMax=customArea[1]
        lonMin=customArea[2]
        lonMax=customArea[3]
    return laMin, laMax, lonMin, lonMax


# ### GenerateTrainingSet(`str`, `list`, `list`, `int`, `int`, `int`, `int`, `int`, `int`, `str`)
# "main" function which launches all the others and save the generated arrays as .npy files. (.npy files store arrays in binary and are readable by `numpy`, processing them is way faster than the usual `panda` datasets and the storing space is significantly shorten.
# 
# The `def` starts a loop that lasts equally to the `dataSetSize` we want. It increments at each loop one day to the absolute starting date, requests Matlab on the new time bound, transforsms the output in arrays, normalizes them and finally stores everyting.
# 
# #### Parameters
# There are 10 initial parameters allowing us to have full control over the generation. Which indice do we want? From which starting date to which ending date? Which are the min/max latitudes and min/max longitudes boundaries? How many days per matrix do we want? How many minutes are needed between each values? And finally, what is the machine learning algorithm we are using for the reconstruction.
# 
# The Machine Learning algorithm parameter has 3 possibilities: 
# 
# * `svr` : for the Support Vector Machine
# - `pr` : for the Polynomial Regression
# - `rfr` : for the Random Forest Regression
# 
# 
# #### Storing format
# We store three arrays. The two first of shape (24,144), 24 degrees in latitude for 144 values each. The third array of shape (12, ) storing all the informations we need for the current day. All arrays gathered we get an array of shape (3, ) symbolising an array of three arrays.
# 
# |Ground truth|Machine learning reconstruction|Infos|
# |:-:|:-:|:-:|
# |Absolute ground truth got from Matlab without any modification|The Ground truth with all nan values filled with ML|Informations about the current matrix|
# |(24,144)|(24,144)|(12, )|
# 
# The information array is constituted of:
# 
# |Date|Max latitude|Min latitude|Max longitude|Min longitude|Max value|Min value|Days|isQuiet|Component type|Working stations|Area|
# |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# |Date of the current matrix stored as datetime object|Latitude degree corresponding to the last index of the array|Latitude degree corresponding to the first index of the array|The maximum longitude degree in which our stations are located|The minimum longitude degree in which are located our stations|Maximum value in the matrix (in nT)|Minimum value in the matrix (in nT)|Number of days in the matrix|Is the current matrix conrresponding to a quiet day?|The component type (x1, x2, y1, y2, alpha etc)|List all working stations name for the given matrix|In which Area are we? (EU, USA, Asia or Custom)|
# |datetime.datetime|Integer|Integer|Integer|Integer|Float|Float|Integer|Bool|String|List|String|
# 
# Note that we have to load these arrays using:
# ```python
# array = np.load("path_to_array/array.npy", allow_pickle=True, encoding="latin1")
# ```
# `allow_pickle=True` and `encoding="latin1"` allows us to load `object` values into `numpy` arrays.

# In[7]:


def GenerateTrainingSet(indice, 
                        startingDate, 
                        endingDate, 
                        area,
                        numberOfDaysPerMatrix, 
                        minutesBetweenValues, 
                        regressor,
                        customArea=[0,0,0,0],
                        interface=False,
                        application=None):

    absoluteStartDate = datetime.datetime(startingDate[0],startingDate[1],startingDate[2],0,0,0)
    absoluteEndDate = datetime.datetime(endingDate[0],endingDate[1],endingDate[2],0,0,0)
    isQuietDay = False
    dataSetSize = absoluteEndDate-absoluteStartDate
    
    for i in range(0,dataSetSize.days):
        EmptyVariables()
        latMin, latMax, longMin, longMax = ChoosePresetArea(area,customArea)
        
        startDateMatrix = absoluteStartDate+datetime.timedelta(days=i)
        startDateMatlab = [startDateMatrix.year, startDateMatrix.month, startDateMatrix.day, 0, 0, 0]
        sys.stdout.flush() # flushes the verbose output allowing us to read everything on time if we launch the script in a nohup subshell
        
        groundTruth = RequestMatlabNew(allStationCodes, allStationLatgeos, allStationLongeos, latMin, latMax, longMin, longMax, startDateMatlab)
        groundTruthNorm = np.empty_like(groundTruth)
        for ilat in range(groundTruth.shape[0]):
            groundTruthNorm[ilat] = normalizeWithGivenBounds(groundTruth[ilat],{'actual': {'lower': np.nanmin(groundTruth), 'upper': np.nanmax(groundTruth)}, 'desired': {'lower': -1, 'upper': 1}})
        

        beforeVariance = groundTruthNorm.copy()
        beforeVariance = RemoveDefectiveStationVariance(beforeVariance)

        ReconstructedArray = PredictIndicatorForAllLatitudes(groundTruthNorm, regressor)
        for a in quietDays:
            compare = np.array(startDateMatlab) == a
            if compare.all(): 
                isQuietDay = True
                break
            else: 
                isQuietDay = False 
                

        infosArray = np.array([startDateMatrix, latMax, latMin, longMax, longMin, np.nanmax(groundTruth), np.nanmin(groundTruth), numberOfDaysPerMatrix, isQuietDay, indice, 'stationsLatitude', area])
        FinalArray = np.array([groundTruth, ReconstructedArray, infosArray])
        np.save("{}/x_train/{}_{}".format(trainingDatasetPathASIA, indice, i), FinalArray)
        print("Matrix saved for date: {}".format(startDateMatrix))
        print("Sample {} out of {}".format(i+1, dataSetSize.days))
        
        if interface:
            totalSamples=float(dataSetSize.days)
            sampleNum=float(i+1)

            UpdateInterface(totalSamples, sampleNum, application, str("Matrix saved for date: {}".format(startDateMatrix)), str("Sample {} out of {}".format(int(sampleNum), int(totalSamples))), int((sampleNum/totalSamples)*100))
            application.update_idletasks()
        clear_output(wait=True)
        del groundTruth, beforeVariance, ReconstructedArray, infosArray, FinalArray


# ### RemoveDefectiveStation(`numpy.array`)
# Takes as input the matrix we are working with and replace by nans all the axis with a `mean_squared_error` too far from the `mean_squared_error` of the mean of all the axis.
# 
# This `def` allows us to considere as non-existant all the stations outputing delusional data. For example, the green line in the middle of the matrix below represents a station outputing zeros for the entire day straight, this `def` will alow us to take it off.
# 
# 
# <img src="Notebook_images/DelusionalDataGreen.jpg" alt="drawing" width="300"/>

# In[8]:


def RemoveDefectiveStationVariance(array):
    for i in range(array.shape[0]):
        if np.var(array[i]) < 0.5: #valeur arbitraire
            array[i] = np.full(array.shape[1], np.nan)
    return array


# ### RequestMatlab()
# Stores all the `indices_alpha` output in a dictionary 

# In[9]:


def RequestMatlabNew(allStationCodes, allStationLatgeos, allStationLongeos, latMin, latMax, longMin, longMax, startDateMatlab):
    stInMat = []
    for i in range(0,allStationCodes.shape[0]): 
        if allStationLatgeos[i]>latMin and allStationLatgeos[i]<latMax and allStationLongeos[i]>longMin and allStationLongeos[i]<longMax:
            stInMat.append(allStationCodes[i])
    stID = []
    stID_latMag = []
    for stCode in stInMat:
        stID.append(stationCodesID.get(stCode))
    err = StringIO.StringIO()
    out = StringIO.StringIO()
    mat,T,LAT = eng.create_matrix(matlab.double(stID),matlab.double(startDateMatlab),'y2','m', latMin, latMax-1, 1./5.99, nargout=3, stderr=err, stdout=out)
    groundTruthMag = np.array(mat)
    return groundTruthMag


# ### IndicatorCalculation(`Dict`, `float`, `datetime.datetime`, `float`)
# Retrieves from `stationsOut` the variation of the indice we chose for a given station in order to create `stationIndicatorVariation` and returns it.
# 
# Returns also the difference as a percentage between the station's magnetic vector magnitude and `igrf`. Allowing us to create our weighting system.
# 
# The weight calculation is implemented as follow:
# $$\frac{\sqrt{x1^2 + y1^2 + z1^2}}{\beta}$$
# with $β$ the magnitude of `igrf` for the given time and station

# In[10]:


def IndicatorCalculation(dataSt, timeshift, currentDate, igrf, indice):
    
    indiceReturn=np.float32(dataSt.get(str(indice))[timeshift])
    
    x1=np.float32(dataSt.get("x1")[timeshift])
    y1=np.float32(dataSt.get("y1")[timeshift])
    z1=np.float32(dataSt.get("z1")[timeshift])
    ratio = ((math.sqrt(pow(x1, 2)+pow(y1, 2)+pow(z1, 2)))/igrf)
    return np.round(indiceReturn,10), ratio


# ### normalizeWithGivenBounds(`numpy.array`, `numpy.array`)
# Normalizes all the values of a vector between the wanted bounds

# In[11]:


def normalizeWithGivenBounds(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]               


# ### PredictIndicatorForAllLatitudesdes(`numpy.array`)
# 
# Main Machine Learning function that triggers the polynomial regression for all longitude degrees.
# 
# We want to train deep learning algorithms on full matrices to test their ability to reconstruct data in controled situations. Therefore, we will be able reproduce the behaviour of the magnetic field in any contextual circumstances. Consequently, the objective here is to do a preliminary reconstruction on highly covered areas like Europe to feed the deep learning algorithms with matrices without any `nan`.
# 
# The `def` takes all the points from all working stations between two latitude bounds, removes the `nan` values for latitudes we don't have data and fits a polynomial regression on the remaining points. The result will be a matrix with the same dimension as the previous one but with all the blank lines filled.
# 
# For example, let's say that we want to fill all `nan` values of the matrix below matrix : 
# 
# <img src="Notebook_images/Before_reconstruction.png" alt="drawing" width="600"/>
# 
# We would have to take this matrix values for every position on the X axis, let's take as example at the red line position :
# 
# <img src="Notebook_images/Before_reconstruction_redLine.png" alt="drawing" width="600"/>
# 
# This "cut" would give us the component's values in regard to the latitude, a 2D representation easy to fit :
# 
# <img src="Notebook_images/ML_processing_0.png" alt="drawing" width="400"/> And after all the blanks filled : <img src="Notebook_images/ML_processing_23.png" alt="drawing" width="400"/>
# 
# Doing this for all latitude values, this process will output this result:
# 
# <img src="Notebook_images/After_reconstruction.png" alt="drawing" width="600"/>

# In[12]:


def PredictIndicatorForAllLatitudes(baseArray, regressor):
    predictionArray = np.empty_like(baseArray)
    predictionArray=np.copy(baseArray)
    RegressorParameters=None
    RegressorParametersPR = {'polynomialfeatures__degree': 2, 'linearregression__fit_intercept': True, 'linearregression__normalize': True} # For PolyRegression
    RegressorParametersSVR = {'kernel' : 'rbf', 'gamma' : 1e-2, 'C' : 10} # For SupportVectorMachinregressorression
    RegressorParametersRFR = {'n_estimators' : 10, 'random_state' : 0} # For RandomForestRegression
    if regressor=='svr': RegressorParameters = RegressorParametersSVR
    elif regressor=='pr': RegressorParameters = RegressorParametersPR
    elif regressor=='rfr': RegressorParameters = RegressorParametersRFR
        
    for i in range(0,baseArray.shape[0]): # for all degrees in latitude
        specificLatitudeTimePrediction = np.full(baseArray.shape[1], np.nan)
        if math.isnan(np.sum(baseArray[i])): # is there any nan in the selected latitude ?
            for y in range(0,baseArray.shape[1]): # for all degrees in longitude
                if (math.isnan(baseArray[i][y])):  # if the current point in the matrix is a nan
                    specificLatitudeTimePrediction[y] = GetIndicatorLongPrediction(i,y,RegressorParameters,predictionArray,regressor) # start the Machine Learning algorithm  
            predictionArray[i] = specificLatitudeTimePrediction # add the predicted values to the global prediction
    return predictionArray


# ### GetIndicatorLongPrediction(`int`,`int`, `dict`, `numpy.array`)
# 
# Gets the point of all stations at a given moment in time, concatenates them and removes the nan values. (ML algorithm don't allow nan values).
# This `def` takes as input a dictionary of parameters, which are hard coded at line 5 of `PredictIndicatorForAllLatitudes`.
# Before, I was doing parameters tuning at each training to fit the data as best as possible. But it appeared through tests that most of the time the parameters resulting from the process were always the same:
# ```python
# {'polynomialfeatures__degree': 2, 'linearregression__fit_intercept': True, 'linearregression__normalize': True}
# ```
# To gain processing time, this functionality is disabled by default. It can be enabled by replacing `RegressorParameters` allocation with the `ParametersTuningPoly(numpy.array,int)` function, which outputs a dictionary of parameters resulted from the tuning.

# In[13]:


def GetIndicatorLongPrediction(latitude,longitude, params, baseArray, regressor):
    indicatorLatVariation = np.array([])
    prediction=None
    for i in range(0, baseArray.shape[0]):
        indicatorLatVariation = np.append(indicatorLatVariation, baseArray[i][longitude])
    y = np.array(indicatorLatVariation)
    x = np.arange(0, baseArray.shape[0], 1)
    x,y = RemoveNan(x, y)
    if regressor=='svr': prediction = SupportVectorMachinregressorression(x,y,params, latitude, longitude).predict(np.array(latitude).reshape(1,-1))
    elif regressor=='pr': prediction = PolyRegression(x,y,params).predict(np.array(latitude).reshape(1,-1))
    elif regressor=='rfr': prediction = RandomForestRegression(x,y,params).predict(np.array(latitude).reshape(1,-1))
       
    return prediction # return the result of the PolyRegression def, defined below


# ### RemoveNan(`numpy.array`, `numpy.array`)
# 
# Takes as input all the indice values corresponding to each latitudes of the matrix, detects where there are nans and remove them.

# In[14]:


def RemoveNan(latValues, indicatorValues):
    indexDeleteY = np.array([])
    for i in range(0, indicatorValues.shape[0]):
        if math.isinf(indicatorValues[i]) or math.isnan(indicatorValues[i]):
            indexDeleteY = np.append(indexDeleteY, i)
    newY = np.delete(indicatorValues, indexDeleteY)
    newX = np.delete(latValues, indexDeleteY)
    newY=newY.reshape(newY.shape[0],1)
    newX=newX.reshape(newY.shape[0],1)
    
    return newX, newY


# ### PolyRegression(`numpy.array`, `numpy.array`, `Dict`)
# 
# Makes a polynomial regression. `poly_grid.fit(X,Y)` where `X` is the latitude and `Y` is the indice. `params` corresponds to the regressor's parameters. 
# The `PolynomialRegression()` definition is custom and detailed below.

# In[15]:


def PolyRegression(latValues, indicatorValues, params):
    poly_grid = PolynomialRegression()
    poly_grid.set_params(**params)
    poly_grid.fit(latValues, indicatorValues)
    return poly_grid


# ### RandomForestRegression(`numpy.array`, `numpy.array`, `Dict`)
# 
# Uses the Sklearn random forest regressor. `RandomForestRegressor.fit(X,Y)` where `X` is the latitude and `Y` is the indice. `params` corresponds to the regressor's parameters.

# In[16]:


def RandomForestRegression(latValues, indicatorValues, params):
    rf = RandomForestRegressor()
    rf.set_params(**params)
    rf.fit(latValues, indicatorValues)
    return rf


# ### SupportVectorMachinregressorression(`numpy.array`, `numpy.array`, `Dict`)
# 
# Uses the Sklearn SVR regressor. `SVR.fit(X,Y)` where `X` is the latitude and `Y` is the indice. `params` corresponds to the regressor's parameters.

# In[17]:


def SupportVectorMachinregressorression(latValues, indicatorValues, params, lat, long):
    svr = SVR()
    svr.set_params(**params)
    svr.fit(latValues, indicatorValues)

    #     The bellow commented code shows two plots as an example of what ML does for each longitude degrees to reconstruct the full matrix

#     if long==38:
#         if lat==0 or lat==23:
#             fig, ax = plt.subplots(1,1)
#             ax.scatter(latValues, indicatorValues, label='Stations values')
#             x = np.arange(24)
#             pred = svr.predict(x.reshape(-1,1)) 
#             ax.plot(x.reshape(-1,1),pred, 'r--', label='Model fit')
#             ax.xaxis.set_ticks(range(0,24,4))
#             ax.xaxis.set_ticklabels(range(30,54,4))
#             ax.legend(loc='best')
#             ax.set_xlabel('Latitude')
#             ax.set_ylabel('Component value')
#             fig.savefig('ML_processing_{}.png'.format(lat))
    return svr


# ### PolynomialRegression(int, **)
# 
# Makes a python pipeline out of `sklearn.preprocessing.PolynomialFeatures` and `sklearn.linear_model.LinearRegression`. This allows us to use a linear regression algorithm on a non-linear fit, giving as parameter the polynom's degree.  

# In[18]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# ### ParametersTuningPoly(`numpy.array`,`int`)
# 
# Makes a quick fit on a given array of data to evaluate the best parameters on the current set. We are testing polynomial degrees from 2 to 5 and check if we should use `linearregression__fit_intercept` or `linearregression__normalize`.

# In[19]:


def ParametersTuningPoly(baseArray,long):
    indicatorLatVariation = np.array([])
    for i in range(0, baseArray.shape[0]):
        indicatorLatVariation = np.append(indicatorLatVariation, baseArray[i][np.int16(long)])
    y = np.array(indicatorLatVariation)
    x = np.arange(0, baseArray.shape[0], 1)
    x, y = RemoveNan(x, y)
    
    paramsTuning = {'polynomialfeatures__degree': [2,5], 'linearregression__fit_intercept': [True, False], 'linearregression__normalize': [True, False]}
    poly_gridTuning = GridSearchCV(PolynomialRegression(), paramsTuning, cv=10, scoring='r2', verbose=0)
    poly_gridTuning.fit(x, y)
    return poly_gridTuning.best_params_


# ### Interface(`object`)
# 
# Create an graphical interface to make the use of the generator more user friendly. It can be triggered by launching the .py script with the `-g` parameter. Typically, the command `python Generate_Training_Dataset.py -g` will start the graphical interface while `python Generate_Training_Dataset.py` won't.

# In[20]:


class Interface(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        self.theme=ttk.Style()
        self.theme.theme_use('clam')
        self.CreateWidgets()
    
    def CreateWidgets(self):
        global app_bg_color
        self.font = Font(family="Comic", size=12)
        self.stickyLabels = 'w'
        self.secondBG = "#ab9c74"
        self.text = tk.Text(self)
        self.text.configure(font=self.font)
        
        
        self.mainFrame = tk.Frame(self, bg=app_bg_color)
        self.mainFrame.pack()
        
        self.parametersFrame = tk.Frame(self.mainFrame, bg=self.secondBG)
        self.parametersFrame.grid(row=0,column=1,pady=20)
        
        self.activeWidgetsFrame = tk.Frame(self.mainFrame, bg=self.secondBG)
        self.activeWidgetsFrame.grid(row=1,column=1)
        
        
        self.component=tk.Label(self.parametersFrame,text="Component", bg=self.secondBG)
        self.component.configure(font=self.font)
        self.component.grid(row=1,column=0, sticky=self.stickyLabels)
        self.component=tk.Entry(self.parametersFrame)
        self.component.insert(tk.END, 'y2')
        self.component.grid(row=1,column=1)

        self.startDate=tk.Label(self.parametersFrame,text="Starting date", bg=self.secondBG)
        self.startDate.configure(font=self.font)
        self.startDate.grid(row=2,column=0, sticky=self.stickyLabels)
        self.startDate=tk.Entry(self.parametersFrame)
        self.startDate.insert(tk.END, '2015,1,1')
        self.startDate.grid(row=2,column=1)

        self.endDate=tk.Label(self.parametersFrame,text="Ending date", bg=self.secondBG)
        self.endDate.configure(font=self.font)
        self.endDate.grid(row=3,column=0, sticky=self.stickyLabels)
        self.endDate=tk.Entry(self.parametersFrame)
        self.endDate.insert(tk.END, '2015,1,6')
        self.endDate.grid(row=3,column=1)

        self.area=tk.Label(self.parametersFrame,text="Area", bg=self.secondBG)
        self.area.configure(font=self.font)
        self.area.grid(row=4,column=0, sticky=self.stickyLabels)
        self.area=tk.Entry(self.parametersFrame)
        self.area.insert(tk.END, 'asia')
        self.area.grid(row=4,column=1)

        self.days=tk.Label(self.parametersFrame,text="Days in the matrix", bg=self.secondBG)
        self.days.configure(font=self.font)
        self.days.grid(row=5,column=0, sticky=self.stickyLabels)
        self.days=tk.Entry(self.parametersFrame)
        self.days.insert(tk.END, '1')
        self.days.grid(row=5,column=1)

        self.minutes=tk.Label(self.parametersFrame,text="Minutes between each value", bg=self.secondBG)
        self.minutes.configure(font=self.font)
        self.minutes.grid(row=6,column=0, sticky=self.stickyLabels)
        self.minutes=tk.Entry(self.parametersFrame)
        self.minutes.insert(tk.END, '10')
        self.minutes.grid(row=6,column=1)

        self.regressor=tk.Label(self.parametersFrame,text="Machine Learning Regressor", bg=self.secondBG)
        self.regressor.configure(font=self.font)
        self.regressor.grid(row=7,column=0, sticky=self.stickyLabels)
        self.regressor=tk.Entry(self.parametersFrame)
        self.regressor.insert(tk.END, 'svr')
        self.regressor.grid(row=7,column=1)
        
        self.launch = tk.Button(self.activeWidgetsFrame, text='Start generator', command= lambda *args : start_GenerateTrainingSet_thread({'indice':self.component.get(), 
                                                                                                                                        'startingDate':np.fromstring(self.startDate.get(), dtype=int, sep=','), 
                                                                                                                                        'endingDate':np.fromstring(self.endDate.get(), dtype=int, sep=','), 
                                                                                                                                        'area':self.area.get(), 
                                                                                                                                        'numberOfDaysPerMatrix':int(self.days.get()), 
                                                                                                                                        'minutesBetweenValues':int(self.minutes.get()), 
                                                                                                                                        'regressor':self.regressor.get(), 
                                                                                                                                        'customArea':[0,0,0,0],
                                                                                                                                        'interface':True,
                                                                                                                                        'application':self}))
        self.launch.grid(row=1, column=0,pady=5)
        
        self.kill = tk.Button(self.activeWidgetsFrame, text='Stop generator', command=self.destroy)
        self.kill.grid(row=1, column=1,pady=5)
        
        self.progressbar = ttk.Progressbar(self.activeWidgetsFrame,orient ="horizontal",length = 200, mode ="determinate")
        self.progressbar.grid(row=2, column=0, columnspan=2)
        self.progressbar["maximum"] = 100
        self.progressbar["value"] = 0
        self.percentageLabel=tk.Label(self.activeWidgetsFrame,text="",bg=self.secondBG)
        self.percentageLabel.grid(row=3, column=0, columnspan=2,pady=10)
        
        self.progressLabel1=tk.Label(self.activeWidgetsFrame,text="",bg=self.secondBG)
        self.progressLabel1.grid(row=4, column=0, columnspan=2)
        self.progressLabel2=tk.Label(self.activeWidgetsFrame,text="",bg=self.secondBG)
        self.progressLabel2.grid(row=5, column=0, columnspan=2)


# ### start_GenerateTrainingSet_thread(Dict)

# In[21]:


def start_GenerateTrainingSet_thread(params):
    th = threading.Thread(target=GenerateTrainingSet, kwargs=params)
    th.start()
#     th.join()


# ### UpdateInterface(`int`, `int`, `object`, `str`, `str`)
# 
# Updates dynamically the interface progress bar and labels through processing.

# In[22]:


def UpdateInterface(maxValue, currentValue, app, pg1, pg2, percent):
    app.progressLabel1['text'] = pg1
    app.progressLabel2['text'] = pg2
    app.progressbar["maximum"]=maxValue
    app.progressbar["value"]=currentValue
    app.percentageLabel['text'] = "{}%".format(percent)



# ### Main Python Script
# 
# This cell launches the main function in a .py script, checking if there is the `-g` (for "graphics") argument at the command's end.
# 
# **Executing this cell in a Notebook won't have the expected behaviour**

# In[ ]:


if len(sys.argv) > 1:
    if sys.argv[1]=='-g':
        app_bg_color = '#180e0c'
        app = Interface()
        app.title("Dataset Generator")
        app.resizable(width=False, height=False)
        appW=400
        appH=350
        posX = (int(app.winfo_screenwidth()) // 2) - (appW // 2)
        posY = (int(app.winfo_screenheight()) // 2) - (appH // 2)
        geo = "{}x{}+{}+{}".format(appW,appH,posX,posY)
        app.geometry(geo)
        app["bg"]=app_bg_color
        app.mainloop()
    else: print("Unknown parameter")
else:
    GenerateTrainingSet(indice='y2', 
            startingDate=[2015,1,1], 
            endingDate=[2015,1,6],
            area='asia',
            numberOfDaysPerMatrix=1, 
            minutesBetweenValues=10, 
            regressor='svr') # launch the main def

