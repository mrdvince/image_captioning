import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from base import BaseModel
from torchvision import models


class ImageCap(BaseModel):
    def __init__(self,  vocab_size,embed_size=300, hidden_size=512, num_layers=1):
        super(ImageCap, self).__init__()
        ## encoder
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        self.hidden_size = hidden_size
        self.embedd = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def features(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    def forward(self, images, captions):
        features = self.features(images)
        print(features.shape)
        batch_size = features.shape[0]
        caption_trimmed = captions[..., :-1]
        embedd = self.embedd(caption_trimmed)
        inputs = torch.cat([features.unsqueeze(1), embed], 1)
        lstm_out, self.hidden = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        tokens = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            out = self.fc(lstm_out.squeeze(1))
            arg_max = out.max(1)
            ind = arg_max[1].item()
            tokens.append(ind)
            inputs = self.embed(arg_max[1].long()).unsqueeze(1)
            if ind == 1:
                break
        return tokens



# class DecoderRNN(BaseModel):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = (hidden_size,)
#         self.embedd = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(
#             embed_size, hidden_size, num_layers=num_layers, batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, vocab_size)
    
#     def forward(self, features, captions):
#         batch_size = features.shape[0]
#         caption_trimmed = captions[..., :-1]
#         embedd = self.embedd(caption_trimmed)
#         inputs = torch.cat([features.unsqueeze(1), embed], 1)
#         lstm_out, self.hidden = self.lstm(inputs)
#         outputs = self.fc(lstm_out)
#         return outputs

#     def sample(self, inputs, states=None, max_len=20):
#         tokens = []
#         for i in range(max_len):
#             lstm_out, states = self.lstm(inputs, states)
#             out = self.fc(lstm_out.squeeze(1))
#             arg_max = out.max(1)
#             ind = arg_max[1].item()
#             tokens.append(ind)
#             inputs = self.embed(arg_max[1].long()).unsqueeze(1)
#             if ind == 1:
#                 break
#         return tokens

print(ImageCap(300,512,300,1))