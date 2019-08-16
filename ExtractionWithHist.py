import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math  
import statistics
from sklearn import preprocessing, svm
from sklearn import neighbors
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from collections import Counter

def convert_2_8bit(img, max_I=0):
    if img.dtype != np.uint8:
        if max_I == 0:
            max_I = np.percentile(img, 99) # Compute the 99th percentile and use that
        conv = 150./max_I # Set max_I to an 8bit ceiling
        img = (img*conv).astype(np.uint8)
        return img
    else:
        print('image is already 8 bits, skipping...')
        return img
#Returns a list of images and a list of labels that are in the same order
def getTrainingData(working_dir):
    img_dir = working_dir + "training_crops\\"
    images = []
    labels = []
    for fi in os.listdir(img_dir):
        defect_type = 0
        ext = fi.split('\\')[-1].split('.')[-1] 
        if ext in {'tif',"tiff"}:
            im = cv2.imread(img_dir+fi, -1)
            im = convert_2_8bit(im)
            
            if "bridge" in fi:
                defect_type = 1
                number = fi.replace("bridge_" , "")
            elif "falsepositive" in fi:
                defect_type = 2
                number = fi.replace("falsepositive_" , "")
            elif "nonwet" in fi:
                defect_type = 2
                number = fi.replace("nonwet_" , "")
            else:
                print("You have some images that don't have a proper category: ", fi);
                
            labels.append(defect_type)
            images.append(im)
    return images, labels

def getTestingData(working_dir):
    img_dir = working_dir + "del_crops\\"
    images = []
    for fi in os.listdir(img_dir):
        ext = fi.split('\\')[-1].split('.')[-1] 
        if ext in {'tif',"tiff"}:
            im = cv2.imread(img_dir+fi, -1)
            #im = convert_2_8bit(im)
            images.append(im)
    return images

def getFeatures(images,scaler = None , detectFunction="SURF", blur=3, preprocess = "Standard Scalar Only"):
    all_features = []
    image_features = []
    bad_pins = []
    if scaler == None:
        thresh_dir = "C:\\Users\\houfu.yan\\Desktop\\FeatureExtraction\\del_thresh_train\\"
    else:
        thresh_dir = "C:\\Users\\houfu.yan\\Desktop\\FeatureExtraction\\del_thresh_test\\"
        
    
    if os.path.exists(thresh_dir):
        print("Directory already exists")
        return -1, -1
    os.mkdir(thresh_dir)    
    img_num = -1
    
    for im in images:
        #Create Mask
        original = im.copy()
        aa, ab = im.shape
        #im = convert_2_8bit(im)
        mask = im.copy()*0
        cv2.circle(mask,(int(aa/2), int(ab/2)), int(aa/4), 1 , -1)
        
        
        #plt.imshow(im, cmap='gray'),plt.show()        
        #Blur image
        if( blur > 0):
            im = cv2.blur(im, (blur,blur))
            #im = cv2.GaussianBlur(im,(5,5),0)
            #im = cv2.medianBlur(im, 5)
        #Get the feature detection function we're using
        if(detectFunction == "ORB"):
            detect = cv2.ORB_create()
        elif(detectFunction == "SIFT"):
            detect = cv2.xfeatures2d.SIFT_create(nfeatures = SIFT_features, contrastThreshold= SIFT_contrast)
        elif(detectFunction == "SURF"):
            detect = cv2.xfeatures2d.SURF_create(400,extended=True)
        else:
            print("Couldn't recognize the function, running SURF")
            detectFunction = "SURF"
            detect = cv2.xfeatures2d.SURF_create(extended=True)
            
        #mas = im.copy()
        
        
        #ratio =  .5 #Higher = less stuff
        #flattened = im.flatten()
        #chunk_mean = np.mean(flattened)
        #sd = np.std(flattened)
        thresh_mask = im.copy()*0
        radius = 25
        cv2.circle(thresh_mask,(int(aa/2), int(ab/2)), radius, 1 , -1)
        thresh_mask = im*thresh_mask
        flat = thresh_mask.flatten()
        img_mean = 1.0*(np.sum(flat)/ ((radius*radius)*3.141592))
        
        im[im<img_mean]=0
        im[im>=img_mean]=255

        img_num +=1
        #Detect features using the mask
        a, b = detect.detectAndCompute(im, mask)
        
        output_img =im*mask
        
        #output_img = cv2.drawKeypoints(output_img,a,None,(255,0,0),4)

        cv2.imwrite(thresh_dir + str(img_num) + ".tif", output_img)        
        try:
            for feature in b:
                all_features.append(feature)
            image_features.append(b)
            #if scaler !=None:
                #im = im*mask
                #plt.imshow(original, cmap = 'gray')
                #plt.show()                                        
                ##im = cv2.drawKeypoints(im,a,None,(255,0,0),4)
                #plt.imshow(im, cmap = 'gray')
                #plt.show()                        
                
                #plt.imshow(original, cmap = 'gray')
                #plt.show()                        
        except:
            im = im*mask
            im = cv2.drawKeypoints(im,a,None,(255,0,0),4)
            plt.imshow(im, cmap = 'gray')
            plt.show()                        
            bad_pins.append(img_num)
            plt.imshow(original, cmap = 'gray')
            plt.show()                        
        

    if scaler == None:
        #Scale features
        scaler = preprocessing.StandardScaler()
        all_features = scaler.fit_transform(np.array(all_features))
        mean = scaler.mean_
        var = scaler.var_
        
        #Performs scaling on image_features, we can't use preprocessing.StandardScaler() here because we have a 3D list that varies in 2D length
        for im in range(len(image_features)):
            for feat in range(len(image_features[im])):
                cur = 0
                for num in range(len(image_features[im][feat])):
                    image_features[im][feat][num] = (image_features[im][feat][num] - mean[cur])/ math.sqrt(var[cur])
                    cur += 1
    else:
        scaler.transform(np.array(all_features))
        mean = scaler.mean_
        var = scaler.var_
        #Performs scaling on image_features, we can't use preprocessing.StandardScaler() here because we have a 3D list that varies in 2D length
        for im in range(len(image_features)):
            for feat in range(len(image_features[im])):
                cur = 0
                for num in range(len(image_features[im][feat])):
                    image_features[im][feat][num] = (image_features[im][feat][num] - mean[cur])/ math.sqrt(var[cur])
                    cur += 1        
    
    
    return all_features, image_features, scaler, bad_pins

#def doOldML(df, real_type):
    #X_train, X_test,y_train, y_test = model_selection.train_test_split(df, real_type, test_size = 0.1)
    
    #clf = SVC(kernel="rbf", C=5, gamma = 'auto')
    #clf.fit(np.array(X_train), np.array(y_train))
    #accuracy = clf.score(X_test, y_test)
    
    #print("------------------------------")
    #print("Acc is: ", accuracy)    
    #return accuracy, clf


def runClustering(all_features, image_features, clus=30, Kmean=None):
    histograms = []
    if Kmean == None:
        Kmean =  KMeans(n_clusters=clus)
        Kmean.fit(all_features)
    
    #Count the features and put them into our histogram
    for im in image_features:
        km = Kmean.predict(im)
        lis = [0]*clus
        for num in km:
            lis[num]+=1
        
        histograms.append(lis)
        
    return histograms, Kmean

def doML(df, real_type):
    #X_train, X_test,y_train, y_test = model_selection.train_test_split(df, real_type, test_size = 0.1)
    #results = []
    clf = SVC(kernel="linear", C=5, gamma = 'auto')
    #clf = SVC(gamma=2, C = 1)
    clf.fit(np.array(df), np.array(real_type))
    #accuracy = clf.score(X_test, y_test)

    #print("------------------------------")
    #print("Acc is: ", accuracy)    
    return clf


#For validation set ------------------------------------------------------------------------------
def createCrops(working_dir, edge_length = 240):
    img_list = []
    img_dir = working_dir + "del_crops\\"
    val_set = working_dir + "validationSet\\"
    pinlocs = []
    pinloc_names = []
    origin = []
    if os.path.exists(img_dir):
        print("Directory already exists")
        return -1, -1
    os.mkdir(img_dir)
    
    corrected_path = val_set + "corrected\\"
    pinloc_path = val_set + "pinlocs\\"
    
    for fi in os.listdir(pinloc_path):
        ext = fi.split('\\')[-1].split('.')[-1] 
        if ext in {'csv',"csv"}:
            pl_df = pd.read_csv(pinloc_path + fi,header=None)
            pinlocs.append(pl_df)
            pinloc_names.append(fi.replace(".csv", ""))
    num = -1
    for cur in range(len(pinloc_names)):
        im = cv2.imread(corrected_path + pinloc_names[cur] + ".tif", -1)
        im = convert_2_8bit(im)
        pin = pinlocs[cur]
        #temp = []
        for row in pin.iterrows():
            y, x = row[1][0], row[1][1]
            origin.append(pinloc_names[cur])
            num +=1
            roi = im[int(x-edge_length/2):int(x+edge_length/2), int(y-edge_length/2): int(y+ edge_length/2) ]
            img_list.append(roi)
            cv2.imwrite(img_dir + str(num) + ".tif", roi)
        #img_list.append(temp)
    
    
    return origin, pinlocs, img_list




#--------------------------------------------------------------------------------------------------

#val_set = "C:\\Users\\houfu.yan\\Desktop\\FeatureExtraction\\validationSet"

def testData(clf, data):
    results = clf.predict(data)
    return results

def outputData(results, working_dir, image_order):
    img_dir = working_dir + "results\\"
    pinlocs = working_dir + "validationSet\\" + "pinlocs\\"
    
    if os.path.exists(img_dir):
        print("Results directory already exists, will not output data")
        return
    
    os.mkdir(img_dir)
    
    
    

working_dir = "C:\\Users\\houfu.yan\\Desktop\\FeatureExtraction\\"
origin, pinlocs, img_list = createCrops(working_dir)
if(origin == -1):
    print("Delete the crop folder and restart")
    exit()
    
    
df = pinlocs[0]
for n in range(0, len(pinlocs)-1):
    df = df.append(pinlocs[n], ignore_index=True)
df.insert(2, column = 2, value=origin)

print(df)
images, labels = getTrainingData(working_dir)
print("We got images")
all_features, image_features, scaler, bad_train_pins = getFeatures(images)
print("We got features")
hist, Kmean= runClustering(all_features, image_features)
print("We done with clustering")
clf = doML(hist, labels)
print("Done with classifier training")
test_imgs = getTestingData(working_dir)
all_data_features, data_image_features, scaler, bad_test_pins = getFeatures(img_list, scaler)
print("Done getting testing features")
testhist, Kmean = runClustering(all_data_features, data_image_features, Kmean=Kmean)
results = testData(clf, testhist)
#outputData(results, working_dir, image_order)
number = 1
for res in results:
    print(number)
    print(res)  
    print("-----------------------------")
        
    number += 1
print("bad pins: ", len(bad_test_pins))
print("bridge: ", results.tolist().count(1))    
print("falsepositive: ", results.tolist().count(2))
print("nonwet: ", results.tolist().count(3))
print("stop here")

#Create pd dataframe
for bad in bad_test_pins:
    results = np.concatenate((results[:bad], [-1], results[bad:]))

df.insert(3, column = 3, value=results)
print(df)
filename = "output.csv"
df.to_csv(filename, index=True)

num = 0
errors = 0
for n in results:
    if num <=29:
        if n != 2:
            errors += 1
    else:
        if n != 1:
            errors += 1
    num +=1

print("Errors are: ", errors)
#high = 0
#low = 10
#a=0
#times = 1000
#all_stuff = []
#classifiers = []
#for _ in range(0, times):
    #num, clf = doML(hist, labels)
    #all_stuff.append(num)
    #classifiers.append(clf)
    #if num > high:
        #high = num
    #if num < low:
        #low = num
    #a += num
#print("-------------------------------")
#print("avg is ", a/times)
#print("high is ", high)
#print("low is ", low)
#print("std is ", statistics.stdev(all_stuff))
#print("-------------------------------")

#print("b is ", b/30)
#print("c is ", c/30)