import h5py
import nibabel as nib
import numpy as np
import os
import math
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from DataAugmentations import preprocess_for_train
import tensorflow as tf

"""
This script generates an arbitrary number of datasets from given directories in volumeDirs and segmentationDirs.
These ith directory in each must contain the volumes and/or segmentations for the ith dataset. 
The number of .nii files in the volume and segmentation directories for each dataset must be equal and the order must be the same (i.e. the first segmentation .nii file and the first volume .nii file must be from the same scan).
"""

volumeDirs = [""] # Enter your path(s) here
segmentationDirs = [""] # Enter your path(s) here

fileNames = [""] # Enter your file name(s) here

# File consistency check for all datasets
for volDir, segmentDir in zip(volumeDirs, segmentationDirs):
    print(f"Checking consistency in directory pair: {volDir} and {segmentDir}")
    
    # List and sort the files to ensure correct pairing
    volumeFiles = sorted(os.listdir(volDir))
    segmentationFiles = sorted(os.listdir(segmentDir))
    
    for volFile, segFile in zip(volumeFiles, segmentationFiles):
        # Ensure files with matching numbers are paired
        if volFile.split('-')[-1] != segFile.split('-')[-1]:
            print(f"Warning: Volume file {volFile} does not match Segmentation file {segFile}")
            continue
        
        volumeScan = nib.load(volDir + volFile)
        segmentation = nib.load(segmentDir + segFile)
        
        volumeData = volumeScan.get_fdata()
        segmentData = segmentation.get_fdata()
        
        if volumeData.shape != segmentData.shape:
            print(f"Mismatch found: Volume file {volFile} has shape {volumeData.shape}, but Segmentation file {segFile} has shape {segmentData.shape}")
        else:
            print(f"Files {volFile} and {segFile} are correctly matched.")

#Percent of slices to keep from each scan, starts from middle of array
keepRate = 0.3

#Resize all slices/segmentations to imageDim x imageDim
imageDim = 256

#Standard window-leveling performed on all slices
def window_level(vol, window_center, window_width): 
    img_min = window_center - window_width // 2 
    img_max = window_center + window_width // 2 
    window_image = vol.copy() 
    window_image[window_image < img_min] = img_min 
    window_image[window_image > img_max] = img_max 

    return window_image 

#Hard-coded maximum and minimum values for full LiTS dataset because recalculating is very slow
minVal = -3055
maxVal = 5811

#Example selection for PolyCL-O
#Selects a random slice with specified label (targetLabel) from all slices in volDir/segmentDir
#Excludes all segmentation/volume files in the excludeFiles list
#Also excludes the current slice, determined by currVolumeName and currSliceNum
#Only tries to randomly select from each file 10 times, then excludes the file and tries again with a different file
#Performs all preprocessing (window-leveling, normalization) within this function

maxRandomIter = 10

def selectSlice(volDir, segmentDir, targetLabel, currVolumeName="", currSliceNum=-1, excludeFiles=[]):
    volumes = sorted(os.listdir(volDir))
    segmentations = sorted(os.listdir(segmentDir))

    for fileName in volumes:
        if fileName[0] == "." or fileName in excludeFiles:
            volumes.remove(fileName)

    for fileName in segmentations:
        if fileName[0] == "." or fileName in excludeFiles:
            segmentations.remove(fileName)

    if len(volumes) == 0 or len(segmentations) == 0:
        return selectSlice(volDir, segmentDir, targetLabel, currVolumeName=currVolumeName, currSliceNum=currSliceNum)

    scanInd = random.randrange(0, len(volumes))

    segmentation = nib.load(segmentDir + segmentations[scanInd])
    segmentData = segmentation.get_fdata()

    volumeScan = nib.load(volDir + volumes[scanInd])
    volumeData = volumeScan.get_fdata()
    volumeData = window_level(volumeData, 40, 400)

    # Calculate bounds for both volume and segmentation data
    start_slice = max(0, int((min(segmentData.shape[2], volumeData.shape[2]) / 2) - (min(segmentData.shape[2], volumeData.shape[2]) / 2 * keepRate)))
    end_slice = min(min(segmentData.shape[2] - 1, volumeData.shape[2] - 1), int((min(segmentData.shape[2], volumeData.shape[2]) / 2) + (min(segmentData.shape[2], volumeData.shape[2]) / 2 * keepRate)))

    sliceInd = random.randrange(start_slice, end_slice + 1)
    sliceInd = min(sliceInd, segmentData.shape[2] - 1)
    sliceInd = min(sliceInd, volumeData.shape[2] - 1)

    randomIter = 0
    while (min(np.amax(segmentData[:,:,sliceInd].astype(np.int16)), 1) != targetLabel or (sliceInd == currSliceNum and volumes[scanInd] == currVolumeName)) and randomIter <= maxRandomIter:
        randomIter += 1
        sliceInd = random.randrange(start_slice, end_slice + 1)
        sliceInd = min(sliceInd, segmentData.shape[2] - 1)
        sliceInd = min(sliceInd, volumeData.shape[2] - 1)

    if randomIter >= maxRandomIter:
        excludeFiles.append(volumes[scanInd])
        excludeFiles.append(segmentations[scanInd])
        return selectSlice(volDir, segmentDir, targetLabel, currVolumeName=currVolumeName, currSliceNum=currSliceNum, excludeFiles=excludeFiles)

    if sliceInd >= 0 and sliceInd < min(segmentData.shape[2], volumeData.shape[2]):
        volumeSlice = np.array(Image.fromarray(volumeData[:,:,sliceInd].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
    else:
        raise ValueError(f"sliceInd {sliceInd} is out of bounds for volume data with shape {volumeData.shape} and segment data with shape {segmentData.shape}.")

    volumeSlice -= float(minVal)
    volumeSlice /= float(maxVal - minVal)

    segmentSlice = segmentData[:,:,sliceInd].astype(np.int16)

    return volumeSlice, segmentSlice, volumes[scanInd]

#Positive and negative example selection processes for PolyCL-S

#Selects random slice from current volume, excluding the current slice
def selectSliceRandPos(volDir, segmentDir, currVolumeName, currSegmentName, currSliceNum):
    segmentation = nib.load(segmentDir + currSegmentName)
    segmentData = segmentation.get_fdata()

    sliceInd = random.randrange(int((segmentData.shape[2] / 2) - (segmentData.shape[2] / 2 * keepRate)), int((segmentData.shape[2] / 2) + (segmentData.shape[2] / 2 * keepRate)))
    while sliceInd == currSliceNum or sliceInd >= segmentData.shape[2]:
        sliceInd = random.randrange(int((segmentData.shape[2] / 2) - (segmentData.shape[2] / 2 * keepRate)), int((segmentData.shape[2] / 2) + (segmentData.shape[2] / 2 * keepRate)))

    volumeScan = nib.load(volDir + currVolumeName)
    volumeData = volumeScan.get_fdata()
    volumeData = window_level(volumeData, 40, 400)
    
    if sliceInd >= volumeData.shape[2]:
        sliceInd = volumeData.shape[2] - 1
        
    volumeSlice = np.array(Image.fromarray(volumeData[:,:,sliceInd].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
    volumeSlice -= float(minVal)
    volumeSlice /= float(maxVal - minVal)

    segmentSlice = segmentData[:,:,sliceInd].astype(np.int16)

    return volumeSlice, segmentSlice, currVolumeName

#First selects random CT scan that's not the current scan
#Then selects random slice from that scan, performs preprocessing and returns it
def selectSliceRandNeg(volDir, segmentDir, currVolumeName):
    volumes = sorted(os.listdir(volDir))
    currVolInd = volumes.index(currVolumeName)

    volInd = random.randrange(0, len(volumes))
    while volInd == currVolInd:
        volInd = random.randrange(0, len(volumes))

    volumeScan = nib.load(volDir + volumes[volInd])
    volumeData = volumeScan.get_fdata()

    sliceInd = random.randrange(int((volumeData.shape[2] / 2) - (volumeData.shape[2] / 2 * keepRate)), int((volumeData.shape[2] / 2) + (volumeData.shape[2] / 2 * keepRate)))

    if sliceInd >= volumeData.shape[2]:
        sliceInd = volumeData.shape[2] - 1
        
    volumeData = window_level(volumeData, 40, 400)
    volumeSlice = np.array(Image.fromarray(volumeData[:,:,sliceInd].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
    volumeSlice -= float(minVal)
    volumeSlice /= float(maxVal - minVal)

    segmentation = nib.load(segmentDir + os.listdir(segmentDir)[volInd])
    segmentData = segmentation.get_fdata()
    
    if sliceInd >= segmentData.shape[2]:
        sliceInd = segmentData.shape[2] - 1
    
    segmentSlice = segmentData[:,:,sliceInd].astype(np.int16)

    return volumeSlice, segmentSlice, volumes[currVolInd]

def selectSlicePolyCLM(volDir, segmentDir, targetLabel, currVolumeName="", currSegmentName="", currSliceNum=-1, excludeFiles=[]):
    segmentation = nib.load(segmentDir + currSegmentName)
    segmentData = segmentation.get_fdata()

    numSlices = segmentData.shape[2]
    print(f"Number of slices in segmentation data: {numSlices}")

    middleIndex = numSlices / 2
    startRange = int(middleIndex - (middleIndex * keepRate))
    endRange = int(middleIndex + (middleIndex * keepRate))

    # Debugging output for slice ranges
    print(f"Calculated middle index: {middleIndex}")
    print(f"Start of slice range: {startRange}")
    print(f"End of slice range: {endRange}")

    sliceInd = random.randrange(startRange, endRange)
    print(f"Selected slice index: {sliceInd}")

    randomIter = 0

    while (min(np.amax(segmentData[:,:,sliceInd].astype(np.int16)), 1) != targetLabel or sliceInd == currSliceNum) and randomIter <= maxRandomIter:
        randomIter += 1
        sliceInd = random.randrange(startRange, endRange)
        print(f"Re-selected slice index (iteration {randomIter}): {sliceInd}")

    if randomIter >= maxRandomIter:
        return None, None, currVolumeName

    volumeScan = nib.load(volDir + currVolumeName)
    volumeData = volumeScan.get_fdata()

    # Cross-check dimensions between segmentation and volume data
    if segmentData.shape[2] != volumeData.shape[2]:
        raise ValueError(f"Mismatch between number of slices in segmentation and volume data: segmentData slices = {segmentData.shape[2]}, volumeData slices = {volumeData.shape[2]}")

    if sliceInd >= volumeData.shape[2]:
        raise IndexError(f"sliceInd {sliceInd} is out of bounds for volumeData with shape {volumeData.shape}")

    volumeData = window_level(volumeData, 40, 400)
    volumeSlice = np.array(Image.fromarray(volumeData[:,:,sliceInd].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
    volumeSlice -= float(minVal)
    volumeSlice /= float(maxVal - minVal)

    segmentSlice = segmentData[:,:,sliceInd].astype(np.int16)

    return volumeSlice, segmentSlice, currVolumeName

def selectNegativeSlicePolyCLM(volDir, segmentDir, targetLabel, currVolumeName="", currSegmentName="", currSliceNum=-1, excludeFiles=[]):
    volumes = sorted(os.listdir(volDir))
    segmentations = sorted(os.listdir(segmentDir))
    currVolInd = volumes.index(currVolumeName)

    # Select a different volume (negative example)
    volInd = random.randrange(0, len(volumes))
    while volInd == currVolInd:
        volInd = random.randrange(0, len(volumes))

    # Load the selected volume and its corresponding segmentation
    volumeScan = nib.load(volDir + volumes[volInd])
    volumeData = volumeScan.get_fdata()

    segmentation = nib.load(segmentDir + segmentations[volInd])  # Ensure correct pairing
    segmentData = segmentation.get_fdata()
    
    # Ensure the slice index is within bounds for the chosen volume and segmentation
    numSlices = volumeData.shape[2]
    sliceInd = random.randrange(int((numSlices / 2) - (numSlices / 2 * keepRate)), int((numSlices / 2) + (numSlices / 2 * keepRate)))

    randomIter = 0
    
    # Ensure the selected slice does not contain the target organ
    while (min(np.amax(segmentData[:,:,sliceInd].astype(np.int16)), 1) == targetLabel) and randomIter <= maxRandomIter:
        randomIter += 1
        sliceInd = random.randrange(int((numSlices / 2) - (numSlices / 2 * keepRate)), int((numSlices / 2) + (numSlices / 2 * keepRate)))
    
    if randomIter >= maxRandomIter:
        print(f"Skipping slice selection in {volumes[volInd]}: could not find a negative slice without the target label.")
        return None, None, None  # Indicate that no valid negative slice was found

    # Check that the slice index is within bounds for the segmentation data
    if sliceInd >= segmentData.shape[2]:
        raise IndexError(f"sliceInd {sliceInd} is out of bounds for segmentData with shape {segmentData.shape}")

    volumeData = window_level(volumeData, 40, 400)
    volumeSlice = np.array(Image.fromarray(volumeData[:,:,sliceInd].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
    volumeSlice -= float(minVal)
    volumeSlice /= float(maxVal - minVal)

    segmentSlice = segmentData[:,:,sliceInd].astype(np.int16)

    return volumeSlice, segmentSlice, volumes[volInd]

#Creates positive example for SimCLR dataset, requires tensorflow to use external code
def simCLRPos(volDir, currVolumeName, currSliceNum):
    volumeScan = nib.load(volDir + currVolumeName)
    volumeData = volumeScan.get_fdata()
    volumeData = window_level(volumeData, 40, 400)
    volumeSlice = np.array(Image.fromarray(volumeData[:,:,currSliceNum].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
    volumeSlice = np.expand_dims(volumeSlice, axis=2)
    volumeSlice = tf.convert_to_tensor(volumeSlice)
    volumeSlice = preprocess_for_train(volumeSlice, 256, 256)
    volumeSlice = volumeSlice.numpy()
    return volumeSlice

#0: PolyCL-O
#1: PolyCL-S
#2: SimCLR
#3: PolyCL-M
datasetType = 2

#Very similar in structure to the fully supervised dataset creation
for i, datasetName in enumerate(fileNames):
    sliceNum = 0

    volumes = sorted(os.listdir(volumeDirs[i]))
    segmentations = sorted(os.listdir(segmentationDirs[i]))

    file = h5py.File(datasetName, 'w')

    for j, volumeName in enumerate(volumes):
        segmentName = segmentations[j]

        volumeScan = nib.load(volumeDirs[i] + volumeName)
        volumeData = volumeScan.get_fdata()
        volumeData = window_level(volumeData, 40, 400)

        segmentation = nib.load(segmentationDirs[i] + segmentName)
        segmentData = segmentation.get_fdata()

        for plane in tqdm(range(
            math.ceil(((volumeData.shape[2] - 1) / 2) - (((volumeData.shape[2] - 1) / 2) * keepRate)), 
            math.floor(((volumeData.shape[2] - 1) / 2) + (((volumeData.shape[2] - 1) / 2) * keepRate))
            )):
            # Ensure that the plane index does not exceed the bounds of both volumeData and segmentData
            if plane >= volumeData.shape[2] or plane >= segmentData.shape[2]:
                continue  # Skip this iteration if plane index is out of bounds
        
            sliceVolume = np.array(Image.fromarray(volumeData[:,:,plane].astype(np.float64)).resize((imageDim, imageDim), Image.BILINEAR))
            sliceVolume -= float(minVal)
            sliceVolume /= float(maxVal - minVal)
        
            sliceSegment = segmentData[:,:,plane].astype(np.int16)
        
            label = min(np.amax(sliceSegment), 1)
        
            # Uses different example selection strategies based on the type of dataset being created
            if datasetType == 0:
                positiveSlice, positiveSegment, positiveScan = selectSlice(volumeDirs[i], segmentationDirs[i], label, currVolumeName=volumeName, currSliceNum=plane)
                negativeSlice, negativeSegment, negativeScan = selectSlice(volumeDirs[i], segmentationDirs[i], 1 - label, currVolumeName=volumeName, currSliceNum=plane)
            elif datasetType == 1:
                positiveSlice, positiveSegment, positiveScan = selectSliceRandPos(volumeDirs[i], segmentationDirs[i], volumeName, segmentName, plane)
                negativeSlice, negativeSegment, negativeScan = selectSliceRandNeg(volumeDirs[i], segmentationDirs[i], volumeName)
            elif datasetType == 2:
                positiveSlice = simCLRPos(volumeDirs[i], volumeName, plane)
            elif datasetType == 3:  # PolyCL-M
                positiveSlice, positiveSegment, positiveScan = selectSlicePolyCLM(volumeDirs[i], segmentationDirs[i], label, currVolumeName=volumeName, currSegmentName=segmentName, currSliceNum=plane)
                if positiveSlice is None:  # If no valid slice is found
                    continue
            
                # Use the new function for selecting a negative example that does not contain the target organ
                negativeSlice, negativeSegment, negativeScan = selectNegativeSlicePolyCLM(volumeDirs[i], segmentationDirs[i], label, currVolumeName=volumeName, currSegmentName=segmentName, currSliceNum=plane)
                
                if negativeSlice is None:  # Skip this iteration if no negative slice was found
                    continue
                    
            currGrp = file.create_group("Slice" + str(sliceNum))
            currGrp.create_dataset("MainSlice", data=sliceVolume)
            currGrp.create_dataset("PositiveSlice", data=positiveSlice)
        
            # Doesn't include any segmentation or negative example data for SimCLR datasets
            if datasetType != 2:
                currGrp.create_dataset("MainSegment", data=sliceSegment)
                currGrp.create_dataset("PositiveSegment", data=positiveSegment)
                currGrp.create_dataset("NegativeSlice", data=negativeSlice)
                currGrp.create_dataset("NegativeSegment", data=negativeSegment)
                currGrp.attrs.create("ImageLabel", label, (1,), "int")
                currGrp.attrs.create("PositiveScan", positiveScan)
                currGrp.attrs.create("NegativeScan", negativeScan)
        
            currGrp.attrs.create("MainScan", volumeName)
        
            sliceNum += 1

        print("Finished scan: " + volumeName)

    print("Finished dataset: " + datasetName)
