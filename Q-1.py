from __future__ import division
from scipy.io import arff
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from scipy.stats import zscore
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Menthod for Z-score normalization.
def normalizeDF(dataFrame):
    dfNormalized = dataFrame.copy()
    colList = list(dataFrame.columns)
    #print(cols)
    for col in range(len(colList)):
        colMean = dataFrame[colList[col]].mean()
        colStd = dataFrame[colList[col]].std()
        #print(col,'= ', colMean)
        #print(col,'= ', colStd)
        if(colStd != 0):
            dfNormalized[colList[col]] = (dataFrame[colList[col]] - colMean)/colStd
    
    return dfNormalized

# Calculate Eculidian Distance. 
def calculateEculidDist(trainDF , testRow):
    #np.sqrt(np.sum(np.square((trainArray - testArray))))
    tmp = (((trainDF.sub( testRow, axis=1))**2).sum(axis=1))**0.5
    #tmp.sort_values(axis=0, ascending=True, inplace=True)
    #print(type(tmp))
    
    return tmp
    
# match if any datapoint changed its cluster.
def isClusterChanged(previousDF , newDF):
    isChanged = False
    diffDF = previousDF[ previousDF['AssigedCluster'] != newDF['AssigedCluster'] ]
    if(len(diffDF) > 0):
        isChanged = True
        
    return isChanged
    

# Loading arff file as dataframe
data = arff.loadarff("segment.arff")
trainDF = pd.DataFrame(data[0])
#trainDF.to_csv("segment.csv")
# Dropping the class column and storing it in a new DF.
classDF = trainDF[['class']].copy()
trainDF.drop('class' , axis=1, inplace=True)

#print(trainDF.head(10))
#print(classDF.head(10))

normalizedTrainDF = normalizeDF(trainDF)
#normalizedTrainDF = normalizedTrainDF.head(10).copy()
#normalizedTrainDF.to_csv("segment-normalized.csv")
#print(normalizedTrainDF.head(10))
#print(normalizedTrainDF)

# Total Number of features 
noOfFeatures = 19
# List of K(Cluster)
kList = [1,2,3,4,5,6,7,8,9,10,11,12]
#kList = [2,3,4]
# Number of Initial Centroid Set
clusteringRun = 25
# Max number of K-mean iteration to find the best centroid
maxIteration = 50
# Initial Centroid indices 
centroidIndicesList = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 105, 261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812, 214,
#centroidIndicesList = [3, 5, 7, 9, 2, 6, 1515, 651, 658, 328, 1160, 108, 422, 88, 105, 261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812, 214,
53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711, 1997, 1378, 827, 1875, 424, 1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 672, 483,
65, 92, 400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908,
1964, 1260, 784, 520, 1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846,
1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641,
661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209, 1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 1485,
945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427, 1434, 953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240,
524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 1250, 613, 152, 1440, 473, 1834, 1387, 1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611,
1776, 123, 979, 1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686, 854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582, 816, 1770, 663, 737,
1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 1472]

sseMeanForEachKValue = []
stdForEachKValue = []
std1ForEachKValue = []
std2ForEachKValue = []
for kValue in kList:
    print("kValue ---" , kValue)
    SSEList = []
    for counter in range(clusteringRun):
        print("Clustering Run :- ", counter)
        initialCentroidIndices = centroidIndicesList[(counter*kValue) : (counter*kValue)+kValue]
        print(initialCentroidIndices)
        initialCentroid2DList = []
        for indices in initialCentroidIndices:
            initialCentroid2DList.append(normalizedTrainDF.iloc[indices-1])
        #print(len(initialCentroid2DList))
        #print(type(initialCentroid2DList[0]))
        #print(initialCentroid2DList)
        prevDFWithAssignedCluster = pd.DataFrame()
        for innerCounter in range(maxIteration):
            #print("Iteration # ---- " , innerCounter)
            tmpDict = {}
            clusterNumber = 1
            prevDistDF = pd.DataFrame()
            for centroid in initialCentroid2DList:
                #print("Dist === ", calculateEculidDist(normalizedTrainDF, centroid))
                tmpDict['DistCluster-'+str(clusterNumber)] = calculateEculidDist(normalizedTrainDF, centroid)
                clusterNumber += 1
            distDF = pd.DataFrame(tmpDict)
            #print(distDF)
            distDF['FinalDist'] = distDF.min(axis=1)
            distDF['AssigedCluster'] = distDF.idxmin(axis=1)
            #print("distDF -----------> ", distDF)
            
            dfWithAssignedCluster = pd.concat([normalizedTrainDF, distDF], axis=1)
            #print(dfWithAssignedCluster.columns.tolist())
            #print(dfWithAssignedCluster)
            # check to see if any data point changes its cluster from the previous run, If not, Break the iteration.
            if not prevDFWithAssignedCluster.empty :
                if((dfWithAssignedCluster['AssigedCluster'] == prevDFWithAssignedCluster['AssigedCluster']).all()):
                    print("********** Breaking the iteration, No change in the cluster. *************** ", innerCounter)
                    break
            prevDFWithAssignedCluster = dfWithAssignedCluster.copy()
            # End
            
            newDFWithMeanCentroid = dfWithAssignedCluster.groupby(['AssigedCluster']).mean()
            #print("newDFWithMeanCentroid ---------> ", newDFWithMeanCentroid)
            
            # Creating the new centroids for the next runs, based on the mean of the cluster.
            initialCentroid2DList = []
            for rowIndex in range(len(newDFWithMeanCentroid)):
                initialCentroid2DList.append(newDFWithMeanCentroid.iloc[rowIndex, 0:noOfFeatures])
            
            #print(len(initialCentroid2DList))
        
        #dfWithAssignedCluster.to_csv("dfWithAssignedCluster"+str(kValue)+".csv")
        #print(dfWithAssignedCluster)
        #print(distDF)
        # To Finf the SSE for each run.
        #sseDF = distDF.copy()
        #print("distDF *************************** " , distDF)
        #sseDF.drop('AssigedCluster' , axis=1, inplace=True)
        sseDF = distDF['FinalDist'].copy()
        sseDF = np.square(sseDF)
        #print("sseDF *************************** " , sseDF)
        print(sseDF.sum())
        SSEList.append(round(sseDF.sum(),2))
        
    print("SSEList *********** ", SSEList)
    
    stdSEE = np.std(SSEList)
    meanSSE = np.mean(SSEList)
    sseMeanForEachKValue.append(meanSSE)
    stdForEachKValue.append( stdSEE )
    std1ForEachKValue.append( (meanSSE - 2*stdSEE) )
    std2ForEachKValue.append( (meanSSE + 2*stdSEE) )
    print("meanSSE --- >", meanSSE)
    print("stdSEE --- >", stdSEE)
    print("stdSEE (1) --- >", meanSSE - 2*stdSEE)
    print("stdSEE (2) --- >", meanSSE + 2*stdSEE)
    
sseDict = {"K-Value" : kList, "SSE(Mean)" : sseMeanForEachKValue , "STD(SSE)" : stdForEachKValue , "(MeanSSE - 2*std)" : std1ForEachKValue,  
          "(MeanSSE + 2*std)" : std2ForEachKValue}
        
print(pd.DataFrame(OrderedDict(sseDict)))

figure = plt.figure()

plt.xlabel('K-Cluster')
plt.ylabel('Cluster SSE')
plt.title('MeanSSE VS K-Cluster')
plt.plot(kList, sseMeanForEachKValue)
plt.errorbar(kList, sseMeanForEachKValue, yerr=2*np.array(stdForEachKValue), fmt="o")
plt.show()

figure.savefig("Plot_Question1.pdf")