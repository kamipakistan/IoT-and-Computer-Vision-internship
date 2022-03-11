
# #########################################################################################
# ################################# Social Distance Detection  ###########################
# ########################################################################################

# Importing libraries
import cv2
import numpy as np
from scipy.spatial import distance as dist
import argparse


# ########################## LOADING FILES FROM THE DESK #######################

# Reading COCO/ object Names file
classesFile = "coco.names"
classNames = []
with open(classesFile, 'r') as f:
    classNames = f.read().splitlines()

# Loading the configuration and weights files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"

# ##############################  Creating YOLO Network ##############################
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def detectPeoples(outputs, img):
    hT, wT, cT = img.shape  # Height , Width, and Channels of image
    result = []
    boundingBoxes = []
    confidenceValues = []
    centroids = []

    # ##### We have multiple outputs from the Yolo network Layers
    # ##### so we will loof through single output one by one
    for output in outputs:
        # #### Now we have 85 elements in each output
        # #### So we will loof through each element one by one

        for detection in output:
            # ## First 5 out of 85 elements in detection are (cx, cy, w, h, conf)
            # ## And the Other 80 are the probability of element having the object in the image
            probScores = detection[5:]  # All the probability score
            classIndex = np.argmax(probScores)  # Index of High probability value
            confidence = probScores[classIndex]  # Getting the exact High prob.. Score through the index

            # If the confidence value of having the particular object in the image > 50
            # then we will save its Bounding boxes and confidence value and its name
            confThreshold = 0.5
            # classIndex ==0 means person Object
            if classIndex == 0 and confidence > confThreshold:
                # The original values are in float so we are converting it to pixels values
                w, h = int(detection[2]*wT), int(detection[3]*hT)

                # These values are the center points not the origin
                # So we are subtracting the img origin from these center pixels to get the origin points
                x, y = int((detection[0]*wT)-w/2), int((detection[1]*hT)-h/2)

                boundingBoxes.append([x, y, w, h])
                confidenceValues.append(float(confidence))
                centroids.append((int(detection[0]*wT), int(detection[1]*hT)))

    # None Maximum Suppression function remove the overlapping boxes on one object
    # And the keep the only one box which have maximum confidence vale
    nmsThreshold = 0.2  # if you are facing the boxes overlapping problem reduce its value
    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)
    # print(indices)
    # Drawing the Bounding boxes on image
    for i in indices:
        i = i[0]
        box = boundingBoxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        r = (confidenceValues[i], (x, y, x + w, y + h), centroids[i])
        result.append(r)
    return result

# Social Distance Measurements Function
def SDM(img, results):
    violate = set()
    if len(results) >= 2:
        # print(f"Results = {results}")
        centroids = np.array([result[2] for result in results])
        # print(centroids.shape)
        D = dist.cdist(centroids, centroids, metric="euclidean")
        print(f"Distance Matrix {D}")
        MinDistance = 60
        for i in range(0, D.shape[0]):
            # print(f"i {i}")
            for j in range(i + 1, D.shape[1]):
                # print(j)
                # print(D[i, j])
                if D[i, j] <= MinDistance:
                    violate.add(i)
                    violate.add(j)

    # Loop over the results
    # confidenceValues, (x, y, x + w, y + h), centroids
    for (i, (prob, bbox, centers)) in enumerate(results):
        (x, y, w, h) = bbox
        # print(f"centers {centers}")
        (cX, cY) = centers
        color = (0, 255, 0)
        if i in violate:
            color = (0, 0, 255)
        cv2.circle(img, (cX, cY), 5, color, 7)
        cv2.rectangle(img, (x, y), (w, h), color, 2)

        # draw the total number of social distancing violations on the
        text = f"Social Distancing Violations: {len(violate)}"
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)


cap = cv2.VideoCapture("E:\Movies\Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4")
whT = 320  # Width and Height of the Image

while cap.isOpened():
    success, img = cap.read()
    # Converting image to blob format
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)
    # Input blob image to the Yolo network
    net.setInput(blob)
    # We need only the names of the output layers
    outputLayersNames = net.getUnconnectedOutLayersNames()
    # Now we will forward the blob image output layers to the network and store its output
    # and from this output we will find the bounding boxes
    outputs = net.forward(outputLayersNames)

    # Calling Detect Peoples Function
    results = detectPeoples(outputs, img)
    # confidenceValues, (x, y, w, h), centroids output of detectPeoples
    SDM(img, results)

    # Displaying the footage
    cv2.imshow('Object Detection', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
