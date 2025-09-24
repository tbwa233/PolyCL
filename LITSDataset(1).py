import h5py
from torch.utils.data import Dataset
import torch
import numpy as np

#Standard binary dataset that can be used for all liver segmentation tasks
class LITSBinaryDataset(Dataset):
    def __init__(self, fileName):
        super().__init__()

        #Keeps a file pointer open throughout use
        self.file = h5py.File(fileName, 'r')

        #Precalculates length to reduce training computations
        self.length = len(list(self.file.keys()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.file["Slice" + str(idx)]["Slice"]
        segmentation = self.file["Slice" + str(idx)]["Segmentation"]
        label = self.file["Slice" + str(idx)].attrs.get("ImageLabel")

        result = []

        #Returns list containing slice data, image label, and segmentation data
        result.append(torch.Tensor(data[...]).unsqueeze(0))
        result.append(torch.clamp(torch.Tensor(segmentation[...]).unsqueeze(0), min=0, max=1))
        result.append(torch.Tensor(label).squeeze(0))

        return result

    def closeFile(self):
        #Closes file once dataset is no longer being used
        #Do not use class instance after this function is called
        self.file.close()

#Standard LiTS multiclass dataset for liver and tumor segmentation
#Similar to the binary dataset, just also returns the tumor segmentation mask
class LITSMultiClassDataset(Dataset):
    def __init__(self, fileName):
        super().__init__()

        #Keeps a file pointer open throughout use
        self.file = h5py.File(fileName, 'r')

        #Precalculates length to reduce training computations
        self.length = len(list(self.file.keys()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.file["Slice" + str(idx)]["Slice"]
        liverSegment = self.file["Slice" + str(idx)]["LiverSegmentation"]
        tumorSegment = self.file["Slice" + str(idx)]["TumorSegmentation"]
        label = self.file["Slice" + str(idx)].attrs.get("ImageLabel")

        result = []

        #Returns list containing slice data and image label, as well as the liver and tumor segmentation maps together
        result.append(torch.Tensor(data[...]).unsqueeze(0))
        result.append(torch.Tensor(np.array([liverSegment[...], tumorSegment[...]])))
        result.append(torch.Tensor(label).squeeze(0))

        return result

    def closeFile(self):
        #Closes file once dataset is no longer being used
        #Do not use class instance after this function is called
        self.file.close()

class LITSContDatasetSimCLR(Dataset):
    def __init__(self, fileName):
        super().__init__()

        #Initiates in the same way as binary and multiclass datasets
        self.file = h5py.File(fileName, 'r')
        self.length = len(list(self.file.keys()))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        #Returns the main and positive slice for SimCLR training
        #Positive slice is a transformed version of the main
        result = []
        result.append(torch.Tensor(self.file["Slice" + str(idx)]["MainSlice"][...]).unsqueeze(0))
        result.append(torch.Tensor(self.file["Slice" + str(idx)]["PositiveSlice"][...]).unsqueeze(0))
        return result
    
    def closeFile(self):
        self.file.close()

#Same class as the SimCLR dataset, just also returns a negative slice for PolyCL training
class LITSContDatasetPolyCL(Dataset):
    def __init__(self, fileName):
        super().__init__()

        self.file = h5py.File(fileName, 'r')
        self.length = len(list(self.file.keys()))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        result = []
        result.append(torch.Tensor(self.file["Slice" + str(idx)]["MainSlice"][...]).unsqueeze(0))
        result.append(torch.Tensor(self.file["Slice" + str(idx)]["PositiveSlice"][...]).unsqueeze(0))
        result.append(torch.Tensor(self.file["Slice" + str(idx)]["NegativeSlice"][...]).unsqueeze(0))
        return result
    
    def closeFile(self):
        self.file.close()