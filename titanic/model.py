"""
Basic MLP classifier for the titanic problem
"""
from torch import nn

class BaseClassifier(nn.Module):
    def __init__(self, in_dim=7, feature_dim=256,  out_dim=2):
        super(BaseClassifier, self).__init__()
        self.classifier = nn.Sequential(
                        nn.Linear(in_dim, feature_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(feature_dim, out_dim, bias=True),
        )
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.classifier(x)
        return logits
