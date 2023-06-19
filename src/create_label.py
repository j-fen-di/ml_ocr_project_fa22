import os
import sys
import csv
import numpy as np
import cv2
import pickle


def createDict():

    currPath = os.path.abspath(os.getcwd())
    upone = os.path.dirname(os.path.dirname(currPath))
    datasetPath = os.path.join(upone, "Team_Bread", "data", "dataset")
    print(datasetPath)

    dirlist = os.listdir(datasetPath)
    if ".DS_Store" in dirlist:
        dirlist.remove(".DS_Store")

    singleChar = [i.lower() for i in dirlist if len(i) < 2]
    multipleChars = [i.lower() for i in dirlist if len(i) > 1]

    singleChar.sort()
    multipleChars.sort()

    dictionary = {}
    counter = 0

    for char in multipleChars:
        dictionary[char] = counter
        counter += 1

    for char in singleChar:
        dictionary[char] = counter
        counter += 1

    file = open("LabelDictionary.csv", "w")
    w = csv.writer(file)

    for key, val in dictionary.items():
        w.writerow([key, val])

    file.close()


createDict()


def loadDictFromCSV(path):
    dict = {}
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 0:
                dict[row[0]] = int(row[1])
    return dict


def createDataset(datasetPath, csvPath):
    dict = loadDictFromCSV(csvPath)
	
    charactersDirectory = os.listdir(datasetPath)

    if ".DS_Store" in charactersDirectory:
        charactersDirectory.remove(".DS_Store")

    file_count = sum([len(files)
                     for root, directory, files in os.walk(datasetPath)])
    counter = 0

    X = np.empty((0, 45, 45), dtype=np.uint8)
    Y = np.empty((0, 1), dtype=np.uint8)

    for charFolder in charactersDirectory:
        folder = os.path.join(datasetPath, charFolder)
        characters_in_folder = os.listdir(folder)

        if ".DS_Store" in characters_in_folder:
            characters_in_folder.remove(".DS_Store")

        charFolder = charFolder.lower()
        for imgName in characters_in_folder:
            imgPath = os.path.join(folder, imgName)

            # read Image
            image = cv2.imread(imgPath)

            # Color to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # reshape
            imageReshape = np.asarray(image).reshape(45, 45)

            X = np.append(X, [imageReshape], axis=0)

            Y = np.append(Y, dict[charFolder])

            counter += 1

            print("Image" + counter  + "of" + "file_count")
    return X, Y


if __name__ == '__main__':
    path = '/Users/rudresh/Documents/School/Semesters/Fall_2022/CS4476/Projects/Team_Bread/data/dataset'
    createDict()

    dict_name = 'LabelDictionary.csv'
    dict = loadDictFromCSV(dict_name)

    X, Y = createDataset(path, dict_name)
    with open('xydata.pickle', 'wb') as f:
        pickle.dump([X, Y], f)
