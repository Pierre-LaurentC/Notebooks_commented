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

# In[1]:


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
from sklearn.preprocessing import PolynomialFeatures
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


# Python allows us to run Matlab functions in background and retrieve their output. The Output format will, of course, be specific, for example, a `float ` output coming out from a Matlab function will be interpreted as `matlab.double` for Linux and `matlab.mlarray.double` for Windows.
# All of these can be reformatted to native python variables. We now can call Matlab functions from the `eng` object. 
# 
# __*Only functions referenced in the installed Matlab's path can be called from the engine*__

# In[2]:


eng = matlab.engine.start_matlab() #starting and storing the matlab engine


# ## Variables instantiation
# The below cell instantiates all the variables we will use in the execution. Working with this amount of `public` variables is not a good practice. But for development purpose and easier debugging time (Jupyter doesn't have any debugger natively), instantiating them as `public` allows us to access them at any time in the process easily. One improvement would be to install an external Debugging plugin to Jupyter-lab and transpose all the `public` variables to `private`.

# In[3]:


quietDays = np.array(eng.quiet()) # we retreive the output of the `quiet()` matlab function, convert it as `numpy.array` and finally store it.

#Instantiate the starting date
year = 2010
month = 1
day = 1
hour = 0
minute = 0
second = 0

# We want 3 days of data retreived, because we are shifting values according to the latitude to create local time matrices, 
# we need one day before and one day after to make enough space for the shift in both directions.

numberOfDaysWithData = 0

# Instantiate the dates, startDateBase corresponds to the absolute starting date, 
# startDate and endDate will be modified through the execution, one day more each epoch. 
startDateMatrix = datetime.datetime(2000,1,1,0,0,0)
endDateMatrix = datetime.datetime(2000,1,1,0,0,0)
absolutstartDate = datetime.datetime(2000,1,1,0,0,0)

# Mandatory to create valid dates for Matlab, the engine only reads list() objects as date input 
startDateMatlab = [startDateMatrix.year, startDateMatrix.month, startDateMatrix.day, startDateMatrix.hour, startDateMatrix.minute, startDateMatrix.second]
endDateMatlab = [startDateMatrix.year, startDateMatrix.month, startDateMatrix.day, startDateMatrix.hour, startDateMatrix.minute, startDateMatrix.second]


stationsWithNoData = [] # list of stations without data for the given date
stationsNonexistentInFolder = [] # if the station loaded in the station.mat file doesn't exist in the folder, we list it here
stationsOut = dict() # contains all the information related to their station name, stores all the output of Matlab's indices_alpha
stationIndicatorVariation = defaultdict(list) # split of stationsOut, contains the values from indices_alpha but for one single indice (the one we chose to work with)

# Setting up the paths (relative path obviously, needs to be changed if the folder structure changes)
trainingDatasetPath = "../TrainingDataset"
trainingDatasetPathASIA = "../TrainingDataset/Asia/"
# Windows : "D:/IRAP/TrainingDataset"
# Linux : "../TrainingDataset"

# load the stations list
# WINDOWS
# mat = loadmat("D:/IRAP/dir.indices/station.mat")
# LINUX
mat = loadmat("../../../../opt/dir.indices/station.mat")

# store the "station" column values from the .mat file 
stationsList = mat.get("station", "none")[0]
allStationCodes=np.array([])
allStationLatgeos=np.array([])
allStationLongeos=np.array([])

# store in separate arrays all the stations code (clf, aae...) and their geographic latitudes
for x in stationsList:
    allStationCodes=np.append(allStationCodes,x[1][0])
    allStationLatgeos=np.append(allStationLatgeos,x[3][0])
    allStationLongeos = np.append(allStationLongeos,x[2][0])

# within which latitude boundaries do we want our matrix
latMin=0
latMax=0

longMin=0
longMax=0

stationsLatitude = [] # all the working stations for the given matrix, with their latitude
stationIndicatorRatioVariation = defaultdict(list) # dictionary assigning to each station name it's weight
numberOfMinutesNeededInTheTimeStamp=0 # the number of minutes within numberOfDaysWithData
timeBetweenValues=0 # how many minutes do we want between each values for one station (increases consequently the computing time)
numberOfValues=0 # how many values retreived for one station
indicatorVariationArray = np.array([]) # the array containing all the magnetic indices for numberOfDaysWithData and each station in UTC
indicatorVariationArrayLocalTime = np.array([]) # the array containing all the magnetic indices for numberOfDaysWithData and each station in local time
normalized01StationIndicatorVariation = defaultdict(list) # same as indicatorVariationArrayLocalTime, but normalized within given bounds
maxValueinDataset=0 # the maximum value in the current matrix
minValueinDataset=0 # the minimum value in the current matrix
ReconstructedArray = np.array([]) # same as normalized01StationIndicatorVariation


# ### EmptyVariables()
# 
# Free all the arrays and dictionaries memory for a fresh loop restart of the following day.

# In[4]:


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

# In[5]:


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

# In[6]:


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

    global startDateMatrix, endDateMatrix, startDateMatlab, endDateMatlab, stationsWithNoData, stationsNonexistentInFolder, stationsOut, indicatorVariationArrayLocalTime, numberOfDaysWithData, latMin,latMax,longMin,longMax, timeBetweenValues, stationsLatitude
    absolutstartDate = datetime.datetime(startingDate[0],startingDate[1],startingDate[2],0,0,0)
    absolutendDate = datetime.datetime(endingDate[0],endingDate[1],endingDate[2],0,0,0)


    
    
    timeBetweenValues=minutesBetweenValues
    numberOfDaysWithData=numberOfDaysPerMatrix+2
    index=0
    isQuietDay = False
    
    dataSetSize = absolutendDate-absolutstartDate
    
    for i in range(0,dataSetSize.days,numberOfDaysPerMatrix):
        EmptyVariables()
        latMin, latMax, longMin, longMax = ChoosePresetArea(area,customArea)
        stationsLatitude = [''] * (latMax-latMin)
        startDateMatrix = absolutstartDate+datetime.timedelta(days=i)
        endDateMatrix = startDateMatrix+datetime.timedelta(days=numberOfDaysWithData)
        startDateMatlab = [startDateMatrix.year, startDateMatrix.month, startDateMatrix.day, 0, 0, 0]
        endDateMatlab = [endDateMatrix.year, endDateMatrix.month, endDateMatrix.day, 0, 0, 0]
        
        sys.stdout.flush() # flushes the verbose output allowing us to read everything on time if we launch the script in a nohup subshell 
        RequestMatlab()
        MakeStationIndicatorVariation(timeBetweenValues=timeBetweenValues, indice=indice)
        ManuallyNormalizeData01()  
        makeIndicatorVariationArray(stationsLatitude, longMin)
        ResizeForPlot()
        beforeVariance = indicatorVariationArrayLocalTime.copy()
        RemoveDefectiveStationVariance(indicatorVariationArrayLocalTime)
        ReconstructedArray = PredictIndicatorForAllLatitudes(indicatorVariationArrayLocalTime, regressor)
        for a in quietDays:
            compare = np.array(startDateMatlab) == a
            if compare.all(): 
                isQuietDay = True
                break
            else: 
                isQuietDay = False 
        infosArray = np.array([startDateMatrix, latMax, latMin, longMax, longMin, maxValueinDataset, minValueinDataset, numberOfDaysPerMatrix, isQuietDay, indice, stationsLatitude, area])
        FinalArray = np.array([indicatorVariationArrayLocalTime, ReconstructedArray, infosArray, beforeVariance])
        np.save("{}/x_train/{}_{}".format(trainingDatasetPathASIA, indice, index), FinalArray)
        print("Matrix saved for date: {}".format(startDateMatrix))
        print("Sample {} out of {}".format((i/numberOfDaysPerMatrix)+1, dataSetSize.days/numberOfDaysPerMatrix))
        
        index+=1
        if interface:
            UpdateInterface(int(dataSetSize.days/numberOfDaysPerMatrix), int(i/numberOfDaysPerMatrix)+1, application, str("Matrix saved for date: {}".format(startDateMatrix)), str("Sample {} out of {}".format((i/numberOfDaysPerMatrix)+1, dataSetSize.days/numberOfDaysPerMatrix)))
            application.update_idletasks()
        clear_output(wait=True)


# ### RemoveDefectiveStation(`numpy.array`)
# Takes as input the matrix we are working with and replace by nans all the axis with a `mean_squared_error` too far from the `mean_squared_error` of the mean of all the axis.
# 
# This `def` allows us to considere as non-existant all the stations outputing delusional data. For example, the green line in the middle of the matrix below represents a station outputing zeros for the entire day straight, this `def` will alow us to take it off.
# 
# 
# <img src="Notebook_images/DelusionalDataGreen.jpg" alt="drawing" width="300"/>

# In[7]:


def RemoveDefectiveStation(array):
    rmseRef = np.array([])
    rmseRefIndex = np.array([])
    for i in range(array.shape[0]):
        if not math.isnan(np.sum(array[i])):
            rmseRefIndex = np.append(rmseRefIndex, i)
            rmseRef = np.append(rmseRef, mean_squared_error(np.nanmean(array, axis=0), array[i]))
    array[np.int16(rmseRefIndex[np.argmax(rmseRef)])] = np.full(array.shape[1], np.nan)
    return array

def RemoveDefectiveStationVariance(array):
    for i in range(array.shape[0]):
        if np.var(array[i]) < 0.5: #valeur arbitraire
            array[i] = np.full(array.shape[1], np.nan)
    return array


# ### RequestMatlab()
# Stores all the `indices_alpha` output in a dictionary 

# In[8]:


def RequestMatlab():
    global startDateMatlab, endDateMatlab, startDateMatrix, endDate, year, month, day, stationsOut
    for i in range(0,allStationCodes.shape[0]): 
        if allStationLatgeos[i]>latMin and allStationLatgeos[i]<latMax and allStationLongeos[i]>longMin and allStationLongeos[i]<longMax:
            try:
                stationsOut[allStationCodes[i]] = eng.indices_alpha(matlab.double(startDateMatlab), matlab.double(endDateMatlab),str(allStationCodes[i]))
            except:
                print('error with {}'.format(startDateMatlab))
                stationsNonexistentInFolder.append(st)


# ### IndicatorCalculation(`Dict`, `float`, `datetime.datetime`, `float`)
# Retrieves from `stationsOut` the variation of the indice we chose for a given station in order to create `stationIndicatorVariation` and returns it.
# 
# Returns also the difference as a percentage between the station's magnetic vector magnitude and `igrf`. Allowing us to create our weighting system.
# 
# The weight calculation is implemented as follow:
# $$\frac{\sqrt{x1^2 + y1^2 + z1^2}}{\beta}$$
# with $β$ the magnitude of `igrf` for the given time and station

# In[9]:


def IndicatorCalculation(dataSt, timeshift, currentDate, igrf, indice):
    
    indiceReturn=np.float32(dataSt.get(str(indice))[timeshift])
    
    x1=np.float32(dataSt.get("x1")[timeshift])
    y1=np.float32(dataSt.get("y1")[timeshift])
    z1=np.float32(dataSt.get("z1")[timeshift])
    ratio = ((math.sqrt(pow(x1, 2)+pow(y1, 2)+pow(z1, 2)))/igrf)
    return np.round(indiceReturn,10), ratio


# ### CalculateIGRF(`Dict`, `datetime.datetime`)
# Requests `matlab.igrf()` and returns the fourth output to get the magnitude of `igrf`
# 
# *Note:* The `nargout=4` in the below code: 
# ```python
# b=eng.igrf(matlab.double([stLongeo]), matlab.double([stLatgeo]), matlab.double([stAlt]), matlab.double([currentDateMatlab]), nargout=4)
# ```
# notifies python that Matlab is going to output 4 results and all of them has to be taken in account. By default, when dealing with multiple outputs, python only stores the last of them. So the `b` variable in the line above will be an array and not a single float.

# In[10]:


def CalculateIGRF(dataSt, currentDate):    
    stLongeo=np.float32(dataSt.get("longeo"))
    stLatgeo=np.float32(dataSt.get("latgeo"))
    stAlt=np.float32(dataSt.get("alt"))
    
    currentDateMatlab = [currentDate.year, currentDate.month, currentDate.day, currentDate.hour, currentDate.minute, currentDate.second]
    b=eng.igrf(matlab.double([stLongeo]), matlab.double([stLatgeo]), matlab.double([stAlt]), matlab.double([currentDateMatlab]), nargout=4)
    return b[3]


# ### MakeStationIndicatorVariation()
# Fills `stationIndicatorVariation` and `stationIndicatorRatioVariation` containing a single indice type and the `igrf` ratio for each station

# In[11]:


def MakeStationIndicatorVariation(timeBetweenValues, indice):
    
    global stationIndicatorVariation
    global stationIndicatorRatioVariation
    global stationIndicatorVariation
    global numberOfMinutesNeededInTheTimeStamp
    global numberOfValues
    global latMin
    global latMax
    
    stationIndicatorRatioVariation = defaultdict(list)
    stationIndicatorVariation.clear()
    numberOfMinutesNeededInTheTimeStamp = 1440*numberOfDaysWithData
    numberOfValues = np.int16(numberOfMinutesNeededInTheTimeStamp/timeBetweenValues)
    for st in stationsOut.keys():
        if stationsOut[st]:
            igrf=CalculateIGRF(stationsOut[st], startDateMatrix)
            delta = endDateMatrix-startDateMatrix
            totalMinutes = (delta.total_seconds()+1)/60
            for i in range(0, np.int16(totalMinutes), timeBetweenValues):
                update = datetime.timedelta(minutes=i)
                currentDate = startDateMatrix+update
                magneticValue, ratio = IndicatorCalculation(stationsOut.get(st), i, currentDate, igrf, indice)
                stationIndicatorVariation[st].append(magneticValue)
                stationIndicatorRatioVariation[st].append(ratio)


# ### normalizeWithGivenBounds(`numpy.array`, `numpy.array`)
# Normalizes all the values of a vector between the wanted bounds

# In[12]:


def normalizeWithGivenBounds(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]               


# ### ManuallyNormalizeData01()
# Applyes the ratio to the values by multiplying `stationIndicatorVariation` with `stationIndicatorRatioVariation`. Just comment lines 37 and 38 to disable the weighting system.
# 
# Uses `normalizeWithGivenBounds` on all the stations contained in `stationIndicatorVariation` to scale the data and store everyting in `normalized01StationIndicatorVariation`
# 
# This `def` fills also the maximum and minimum values in the matrix before rescaling everything, allowing us to rescale them to their default values when nedeed. (does the same process for the weights)

# In[13]:


def ManuallyNormalizeData01():
    global normalized01StationIndicatorVariation
    global maxValueinDataset
    global minValueinDataset
    normalized01StationIndicatorVariation = defaultdict(list)
    maxValueinDataset=0
    minValueinDataset=0
    max_values = np.array([])
    min_values = np.array([])
    max_values_ratio = np.array([])
    min_values_ratio = np.array([])
    
    for st in stationsOut.keys():
        if stationIndicatorVariation[st]:
            if not math.isnan(stationIndicatorVariation[st][0]):
                max_values = np.append(max_values, max(stationIndicatorVariation[st])) 
                min_values = np.append(min_values, min(stationIndicatorVariation[st]))
                max_values_ratio = np.append(max_values_ratio, max(stationIndicatorRatioVariation[st])) 
                min_values_ratio = np.append(min_values_ratio, min(stationIndicatorRatioVariation[st])) 
    totalMax = max(max_values)
    totalMin = min(min_values)
    totalMaxRatio = max(max_values_ratio)
    totalMinRatio = min(min_values_ratio)
    maxValueinDataset = totalMax
    minValueinDataset = totalMin
    
    bounds = np.array([0,1])   
    boundsRatio = np.array([-1,1])   
    for st in stationsOut.keys():
        if stationIndicatorVariation[st]:
            localMax = max(stationIndicatorVariation[st])
            localMin = min(stationIndicatorVariation[st])
            localMaxRatio = max(stationIndicatorRatioVariation[st])
            localMinRatio = min(stationIndicatorRatioVariation[st])
            
            IndicatorVariationAppliedRatio=stationIndicatorVariation.copy()
#             for i in range(0, len(stationIndicatorVariation[st])):
#                 IndicatorVariationAppliedRatio[st][i] = stationIndicatorVariation[st][i]*stationIndicatorRatioVariation[st][i]
            normalized01StationIndicatorVariation[st] = normalizeWithGivenBounds(np.array(IndicatorVariationAppliedRatio[st]), {'actual': {'lower': totalMin, 'upper': totalMax}, 'desired': {'lower': bounds[0], 'upper': bounds[1]}})


# ### indexValueOnLocalTime(`numpy.array`, `str`, `int`)
# Converts a station's longitude in minutes (assuming 1 degree = 4 minutes) related to Greenwich and shifts back everything to the reference to have arrays representing data in local time.
# 
# Example of the effect of `indexValueOnLocalTime` on `stationName = clf` (not far away from Greenwich):
# 
# <img src="Notebook_images/VariationUTC_LocalTime.png" alt="drawing" width="800"/>

# In[14]:


def indexValueOnLocalTime(array, stationName, i):
    numberOfValuesLong = array.shape[1]
    localTimeValuesArray = np.full((numberOfValuesLong), np.nan)
    long = float(stationsOut[stationName].get("longeo"))
    shiftValues = np.round((long*4)/timeBetweenValues,0)
    initialShiftValues = shiftValues
    increasingIndex=0
    for y in range(np.int16(numberOfValues/numberOfDaysWithData),numberOfValues):
        localTimeValuesArray[increasingIndex] = array[i][np.int16(y-shiftValues)]
        increasingIndex+=1
            
    return localTimeValuesArray 


# In[15]:


def indexValueOnLocalTimeRefLong(array, stationName, i, refLong):
    numberOfValuesLong = array.shape[1]
    localTimeValuesArray = np.full((numberOfValuesLong), np.nan)
    long = float(stationsOut[stationName].get("longeo"))
    shiftValues = np.round(((long-refLong)*4)/timeBetweenValues,0)
    increasingIndex=0
    for y in range(np.int16(numberOfValues/numberOfDaysWithData),numberOfValues):
        localTimeValuesArray[increasingIndex] = array[i][np.int16(y-shiftValues)]
        increasingIndex+=1
            
    return localTimeValuesArray 


# ### makeIndicatorVariationArray()
# Fills `indicatorVariationArray` (UTC) and `indicatorVariationArrayLocalTime` (LT) which are the final versions we will contruct the dataSet on. They are `numpy.array` each rows corresponding to a latitude included between `latMax` and `latMin` with an axis=0 lenght equals to 1440/`timeBetweenValues` (1440 being the number of minutes in one day). in the current state of the art the shape of the arrays is (24,144), 24 = `latMax` - `latMin` and 144 = 1440 / `timeBetweenValues` with timeBetweenValues = 10 and 1440 = the number of minutes in 24h.
# 
# __When the `def` encounters two stations located at the same latitude, it automatically overwrite a station with the closest one to Greenwich.__ 

# In[16]:


def makeIndicatorVariationArray(stationLatitude, minLong):
    global indicatorVariationArray
    global indicatorVariationArrayLocalTime
      
    indicatorVariationArray = np.full((latMax-latMin, len(normalized01StationIndicatorVariation[list(normalized01StationIndicatorVariation.keys())[1]])), np.nan)
    localIndicatorVariationArray = np.full_like(indicatorVariationArray, np.nan)
    indicatorVariationArrayLocalTime = np.full((latMax-latMin, len(normalized01StationIndicatorVariation[list(normalized01StationIndicatorVariation.keys())[1]])), np.nan)
    localNormalized01StationIndicatorVariation = np.full((latMax-latMin, len(normalized01StationIndicatorVariation[list(normalized01StationIndicatorVariation.keys())[1]])), np.nan)
    
    stationsPerLat = defaultdict(list)
    intermediateLocalIndicatorVariationArray = np.empty_like(localIndicatorVariationArray)
    intermediateLocalNormalized01StationIndicatorVariation = np.empty_like(localNormalized01StationIndicatorVariation)
    for st in stationsOut.keys():
        for i in range(latMin, latMax):
            if not isinstance(stationsOut[st], matlab.double):
                if i == np.round(np.int16(stationsOut[st].get("latgeo")),0):
                    stationsPerLat[i-latMin].append(st)
                    if len(stationIndicatorVariation[st])!=0:
                        indicatorVariationArray[i-latMin]=stationIndicatorVariation[st]
                        indicatorVariationArrayLocalTime[i-latMin] = indexValueOnLocalTime(indicatorVariationArray, st, i-latMin)
                        stationLatitude[i-latMin] = st
                    else:
                        None
                else:
                    None


# ### ResizeForPlot()
# Because we are using one more day to the right and to the left to have enough space for `indexValueOnLocalTime()`, this `def` cuts the additional days to keep the one we are interrested in.

# In[17]:


def ResizeForPlot():
    global indicatorVariationArrayLocalTime
    global indicatorVariationArray
    indicatorVariationArrayResized = np.empty([indicatorVariationArray.shape[0], np.int16(numberOfValues-((numberOfValues/numberOfDaysWithData)*2))])
    indicatorVariationArrayLocalTimeResized = np.empty([indicatorVariationArrayLocalTime.shape[0], np.int16(numberOfValues-((numberOfValues/numberOfDaysWithData)*2))])
    m=0
    for i in range(indicatorVariationArray.shape[0]):
        a=0
        for y in range(np.int16(numberOfValues/numberOfDaysWithData),np.int16(numberOfValues-(numberOfValues/numberOfDaysWithData))):
            indicatorVariationArrayResized[m][a]=indicatorVariationArray[i][y]
            a+=1
        m+=1
    indicatorVariationArray = np.empty_like(indicatorVariationArrayResized)
    indicatorVariationArray=indicatorVariationArrayResized[:]
    m=0
    for i in range(indicatorVariationArrayLocalTime.shape[0]):
        a=0
        for y in range(np.int16(numberOfValues/numberOfDaysWithData),np.int16(numberOfValues-(numberOfValues/numberOfDaysWithData))):
            indicatorVariationArrayLocalTimeResized[m][a]=indicatorVariationArrayLocalTime[i][y]
            a+=1
        m+=1
    indicatorVariationArrayLocalTime = np.empty_like(indicatorVariationArrayLocalTimeResized)
    indicatorVariationArrayLocalTime=indicatorVariationArrayLocalTimeResized[:]


# ### PredictIndicatorForAllLatitudesdes(`numpy.array`)
# 
# Main Machine Learning function that triggers the polynomial regression for all longitude degrees.
# 
# We want to train deep learning algorithms on full matrices to test their ability to reconstruct data in controled situations. Therefore, we will be able reproduce the behaviour of the magnetic field in any contextual circumstances. Consequently, the objective here is to do a preliminary reconstruction on highly covered areas like Europe to feed the deep learning algorithms with matrices without any `nan`.
# 
# The `def` takes all the points from all working stations between two latitude bounds, removes the `nan` values for latitudes we don't have data and fits a polynomial regression on the remaining points. The result will be a matrix with the same dimension as the previous one but with all the blank lines filled.
# 
# The below plot shows an example of retrieving all the stations point for a given longitude and a polynomial fit to fill any blank area:
# 
# <img src="Notebook_images/figPrintModelFitForGivenLong.png" alt="drawing" width="800"/>
# 
# This process allows us to make a solid preliminary reconstruction like the one below:
# 
# Starting from this matrix : <img src="Notebook_images/groundTruth.png" alt="drawing" width="300"/> the reconstruction outputs this result: <img src="Notebook_images/groundTruthML.png" alt="drawing" width="300"/>

# In[18]:


def PredictIndicatorForAllLatitudes(baseArray, regressor):
    latsWithoutData = np.array([])
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
            latsWithoutData = np.append(latsWithoutData, i+latMin) # if yes, add it to the empty latitudes list 
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

# In[19]:


def GetIndicatorLongPrediction(latitude,longitude, params, baseArray, regressor):
    indicatorLatVariation = np.array([])
    prediction=None
    for i in range(0, baseArray.shape[0]):
        indicatorLatVariation = np.append(indicatorLatVariation, baseArray[i][longitude])
    y = np.array(indicatorLatVariation)
    x = np.arange(0, baseArray.shape[0], 1)
    x,y = RemoveNan(x, y)
    
    if regressor=='svr': prediction = SupportVectorMachinregressorression(x,y,params).predict(np.array(latitude).reshape(1,-1))
    elif regressor=='pr': prediction = PolyRegression(x,y,params).predict(np.array(latitude).reshape(1,-1))
    elif regressor=='rfr': prediction = RandomForestRegression(x,y,params).predict(np.array(latitude).reshape(1,-1))
        
    return prediction # return the result of the PolyRegression def, defined below


# ### RemoveNan(`numpy.array`, `numpy.array`)
# 
# Takes as input all the indice values corresponding to each latitudes of the matrix, detects where there are nans and remove them.

# In[20]:


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

# In[21]:


def PolyRegression(latValues, indicatorValues, params):
    poly_grid = PolynomialRegression()
    poly_grid.set_params(**params)
    poly_grid.fit(latValues, indicatorValues)
    return poly_grid


# ### RandomForestRegression(`numpy.array`, `numpy.array`, `Dict`)
# 
# Uses the Sklearn random forest regressor. `RandomForestRegressor.fit(X,Y)` where `X` is the latitude and `Y` is the indice. `params` corresponds to the regressor's parameters.

# In[22]:


def RandomForestRegression(latValues, indicatorValues, params):
    rf = RandomForestRegressor()
    rf.set_params(**params)
    rf.fit(latValues, indicatorValues)
    return rf


# ### SupportVectorMachinregressorression(`numpy.array`, `numpy.array`, `Dict`)
# 
# Uses the Sklearn SVR regressor. `SVR.fit(X,Y)` where `X` is the latitude and `Y` is the indice. `params` corresponds to the regressor's parameters.

# In[23]:


def SupportVectorMachinregressorression(latValues, indicatorValues, params):
    svr = SVR()
    svr.set_params(**params)
    svr.fit(latValues, indicatorValues)
    return svr


# ### PolynomialRegression(int, **)
# 
# Makes a python pipeline out of `sklearn.preprocessing.PolynomialFeatures` and `sklearn.linear_model.LinearRegression`. This allows us to use a linear regression algorithm on a non-linear fit, giving as parameter the polynom's degree.  

# In[24]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# ### ParametersTuningPoly(`numpy.array`,`int`)
# 
# Makes a quick fit on a given array of data to evaluate the best parameters on the current set. We are testing polynomial degrees from 2 to 5 and check if we should use `linearregression__fit_intercept` or `linearregression__normalize`.

# In[25]:


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

# In[26]:


class Interface(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        self.theme=ttk.Style()
        self.theme.theme_use('clam')
        self.CreateWidgets()
    
    def CreateWidgets(self):
        global app_bg_color
        self.font = Font(family="Comic", size=12, weight="bold", slant="italic")
        self.stickyLabels = 'w'
        self.secondBG = "#b66d38"
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
        self.progressbar.grid(row=2, column=0,columnspan=2)
        self.progressbar["maximum"] = 100
        self.progressbar["value"] = 0
        
        
        self.progressLabel1=tk.Label(self.activeWidgetsFrame,text="")
        self.progressLabel1.grid(row=3, column=0, columnspan=2)
        self.progressLabel2=tk.Label(self.activeWidgetsFrame,text="")
        self.progressLabel2.grid(row=4, column=0, columnspan=2)


# ### start_GenerateTrainingSet_thread(Dict)

# In[28]:


def start_GenerateTrainingSet_thread(params):
    th = threading.Thread(target=GenerateTrainingSet, kwargs=params)
    th.start()


# ### UpdateInterface(`int`, `int`, `object`, `str`, `str`)
# 
# Updates dynamically the interface progress bar and labels through processing.

# In[29]:


def UpdateInterface(maxValue, currentValue, app, pg1, pg2):
    app.progressLabel1['text'] = pg1
    app.progressLabel2['text'] = pg2
    app.progressbar["maximum"]=maxValue
    app.progressbar["value"]=currentValue
    



if len(sys.argv) > 1:
    if sys.argv[1]=='-g':
        app_bg_color = '#180e0c'
        app = Interface()
        app.title("Dataset Generator")
        app.resizable(width=False, height=False)
        appW=400
        appH=260
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

