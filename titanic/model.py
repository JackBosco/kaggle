"""
Basic MLP classifier for the titanic problem
"""
from torch import nn, transpose

class BaseClassifier(nn.Module):
    def __init__(self, in_dim=22, feature_dim=256, out_dim=2):
        super(BaseClassifier, self).__init__()
        self.classifier = nn.Sequential(
                        nn.Linear(in_dim, feature_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(feature_dim, feature_dim, bias=True),
                        nn.Dropout(),
                        nn.ReLU(),
                        nn.Linear(feature_dim, out_dim, bias=True)
        )
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.classifier(x)
        return logits

class ReverseClassifier(nn.Module):
    def __init__(self, in_dim=2, feature_dim=256,  out_dim=7):
        super(ReverseClassifier, self).__init__()
        self.classifier = nn.Sequential(
                        transpose(nn.Linear(in_dim, feature_dim, bias=True), 0, 1),
                        nn.ReLU(),
                        transpose(nn.Linear(feature_dim, out_dim, bias=True), 0, 1),
        )
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.classifier(x)
        return logits
