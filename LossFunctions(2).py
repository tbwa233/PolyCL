import torch
from torch import nn
import math
from skimage import measure
from sklearn.metrics import f1_score
from torch.nn import functional as F
import numpy as np

#Standard cross-entropy loss for binary training with weights for each class
class BalancedCELoss(nn.Module):
    def __init__(self, weight0=1, weight1=1):
        super().__init__()

        self.weight0 = weight0
        self.weight1 = weight1

    def forward(self, input, target):
        loss = 0

        #Takes negative average loss over each element of input
        #Loss = ln(prediction) * class weight
        #prediction is the predicted likelihood that the correct label is true
        for i, el in enumerate(input):
            if target[i] == 1:
                loss += torch.log(el) * self.weight1
            else:
                loss += torch.log(1 - el) * self.weight0

        return -1 * loss / len(input)

#Focal loss function published at https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, weight0=1, weight1=1, gamma=0):
        super().__init__()

        self.weight0 = weight0
        self.weight1 = weight1
        self.gamma = gamma

    def forward(self, input, target):
        loss = 0

        #Takes negative average loss over each element of input
        #Loss = ln(prediction) * (absolute loss ^ gamma) * class weight
        #prediction is the predicted likelihood that the correct label is true
        for i, el in enumerate(input):
            if target[i] == 1:
                loss += torch.log(el) * (abs(1 - el) ** self.gamma) * self.weight1
            else:
                loss += torch.log(1 - el) * (abs(0 - el) ** self.gamma) * self.weight0

        return (-1 * loss / len(input)).squeeze(0)
     
#Cosine similarity-based loss function
#Modified version of the one used in the SimCLR paper where only one negative example exists
class ContrastiveLossCosine(nn.Module):
    def __init__(self, temp):
        super().__init__()
        
        self.temp = temp

    def forward(self, pred, positive, negative):
        cos = nn.CosineSimilarity(dim=1)
        cosPos = torch.exp(cos(pred, positive) / self.temp)
        cosNeg = torch.exp(cos(pred, negative) / self.temp)

        result = cosPos / (cosPos + cosNeg)
        result = -1 * torch.log(result)

        return result
    
#Contrastive loss function as defined in SimCLR paper https://arxiv.org/abs/2002.05709
class ContrastiveLossSimCLR(nn.Module):
    def __init__(self, temp, device):
        super().__init__()
        
        self.temp = temp
        self.device = device

    def forward(self, pred, positive):
        cos = nn.CosineSimilarity(dim=0)
        result = 0
        for i, anch in enumerate(pred):
            cosPos = torch.exp(cos(anch, positive[i]) / self.temp)
            negSum = 0
            
            for j, anch2 in enumerate(pred):
                if i == j:
                    continue
                
                negSum += torch.exp(cos(anch, anch2) / self.temp)

            for neg in positive:
                negSum += torch.exp(cos(anch, neg) / self.temp)

            curr = cosPos / negSum
            curr = -1 * torch.log(curr)

            result += curr

        return torch.Tensor(result / pred.size(dim=0))

#Uses the euclidean distance from the anchor projection to its positive example and its negative example
#Performs sigmoid on the result to normalize the values
def ContrastiveLossEuclidean(pred, positive, negative):
    posDist = (positive - pred).pow(2)
    while len(posDist.size()) > 1:
        posDist = posDist.sum(-1)
    posDist = torch.sigmoid(posDist.sqrt())

    negDist = (negative - pred).pow(2)
    while len(negDist.size()) > 1:
        negDist = negDist.sum(-1)
    negDist = torch.sigmoid(negDist.sqrt())

    return (1 - posDist) + negDist

#Standard classification accuracy
def accuracy(input, target):
    predictions = torch.round(input)
    accuratePreds = 0

    #Takes negative average loss over each element of input
    #Loss = ln(prediction) * class weight
    #prediction is the predicted likelihood that the correct label is true
    for i, el in enumerate(input):
        if predictions[i] == target[i]:
            accuratePreds += 1

    return accuratePreds / input.size()[0]

#Dice loss for multiclass segmentation where one class is weighted differently than the other
#In future update, will make this into a class like Balanced CE so that the weights can be easily stored when the loss function is first referenced
def weighted_dice_loss(pred, target, smooth = 1., weights=torch.Tensor([1, 5])):
    dim = pred.size()[2]
    weights = weights.to(target.get_device())

    #slice, channel, height, width
    if dim != target.size()[2]:
        target = F.interpolate(target, size=int(dim))

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred.mul(target)).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    loss = loss.mean(dim=0)

    finalLoss = torch.mul(loss, weights).sum()
    return finalLoss

#Uses PyTorch pixel-wise CE loss
def binary_pixel_ce(pred, target):
    if pred.size()[1] > 1 or target.size()[1] > 1:
        print("Warning: only used for binary CE loss, not multiclass")

    return torch.binary_cross_entropy_with_logits(pred, target).mean()

#Standard dice loss, taken directly from MultiMix paper
def dice_loss(pred, target, smooth = 1.):
    dim = pred.size()[2]
    
    #Resizes target if given different sized images
    #pred/target dimensions: slice, channel, height, width
    if dim != target.size()[2]:
        target = F.interpolate(target, size=int(dim))

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred.mul(target)).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
    
#Standard dice score, same as dice loss but rounds predictions first
def dice_score(pred, target, smooth = 1.):
    roundedPreds = torch.round(pred)

    intersectionRounded = (roundedPreds.mul(target)).sum(dim=2).sum(dim=2)
    roundedLoss = ((2. * intersectionRounded + smooth) / (roundedPreds.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))

    return roundedLoss.mean()

#Hausdorff distance, finds closest distance from every point on the predicted structure's contour to a point on the target structure's contour
#Because Hausdorff distance is undefined if either the prediction's or target's structure doesn't exist, these cases are not counted
#Returns separate average scores for each class in a multiclass scenario
#tens1 = predictions, tens2 = target
def hausdorff(tens1, tens2):
    result = np.array([0 for _ in range(tens1.size(dim=1))])
    numMaps = np.array([0 for _ in range(tens1.size(dim=1))])

    #Loops through every slice and class channel
    for i in range(tens1.size(dim=0)):
        for mapChannel in range(tens1.size(dim=1)):
            map1 = tens1[i][mapChannel]
            map2 = tens2[i][mapChannel]

            #Skips slice if the target doesn't contain liver
            #This doesn't work for the predictions because the values aren't rounded
            if torch.count_nonzero(map2) == 0:
                continue

            #Finds the edges of each structure
            cont1 = measure.find_contours(map1.detach().cpu().numpy(), 0.9)
            cont2 = measure.find_contours(map2.detach().cpu().numpy(), 0.9)

            #Finds the maximum minimum distance from any point in cont1 to a point in cont2
            currMax = 0
            numPts = 0
            for line1 in cont1:
                for point1 in line1:
                    numPts += 1
                    minDist = float('inf')
                    for line2 in cont2:
                        for point2 in line2:
                            minDist = min(minDist, dist(point1, point2))

                    currMax = max(currMax, minDist)

            #Adds the distance to the current total for average calculation
            #currMax is 0 if no liver is predicted
            result[mapChannel] += currMax

            #Adds to the denominator in the average calulcation if both maps contained/predicted liver
            if numPts > 0:
                numMaps[mapChannel] += 1

    #Sets the return value to -1 if none of the predictions/ground truth slices contained liver so the training code can sense it
    #The training code will disregard any negative values so it doesn't impact the average loss calculation there
    for i, el in enumerate(numMaps):
        if el == 0:
            result[i] = -1
            numMaps[i] = 1

    #Returns average
    return np.divide(result, numMaps)

#Used in Hausdorff calculation above, just Euclidean distance
def dist(p1, p2):
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

#Returns standard f1 score but sets it to -1 if f1 is undefined, which will be ignored by the evaluation code
def f1(pred, target):
    try:
        return f1_score(target.detach().cpu().numpy(), torch.round(pred).squeeze(1).detach().cpu().numpy(), zero_division=0)
    except:
        return -1