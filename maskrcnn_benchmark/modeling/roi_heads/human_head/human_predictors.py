import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling import registry

@registry.ROI_HUMAN_PREDICTORS.register("HumanCentricPredictor")
class HumanCentricPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        num_actions = cfg.MODEL.ROI_HUMAN_HEAD.NUM_ACTIONS
        super(HumanCentricPredictor, self).__init__()
        
        num_inputs = in_channels

        self.action = nn.Linear(num_inputs, num_actions)
        self.target = nn.Linear(num_inputs, num_actions * 4)

        for l in [self.action, self.target]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        action_logit = self.action(x)
        target_pred = self.target(x)
        return action_logit, target_pred

def make_human_predictor(cfg, in_channels):
    return HumanCentricPredictor(cfg, in_channels)
