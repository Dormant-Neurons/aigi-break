from . import clip
import torch.nn as nn
import torch


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()
        if "Robust-eps4" in name:
            name = name[:-12]
            self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
            checkpoint = torch.load('models/robust_clip/fare_eps_4.pt', map_location=torch.device('cpu'))
            self.model.visual.load_state_dict(checkpoint)
        elif "Robust-eps2" in name:
            name = name[:-12]
            self.model, self.preprocess = clip.load(name, device="cpu")
            checkpoint = torch.load('models/robust_clip/fare_eps_2.pt', map_location=torch.device('cpu'))
            self.model.visual.load_state_dict(checkpoint)
        else:
            self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)

