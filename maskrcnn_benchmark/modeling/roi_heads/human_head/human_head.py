import torch
from torch import nn
from torch.nn import functional as F

from .human_feature_extractors import make_human_feature_extractor
from .human_predictors import make_human_predictor
from .inference import make_human_post_processor
from .loss import make_human_loss_evaluator


class HumanCentricHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(HumanCentricHead, self).__init__()
        self.feature_extractor = make_human_feature_extractor(cfg, in_channels)

        self.predictor = make_human_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_human_post_processor(cfg)
        self.loss_evaluator = make_human_loss_evaluator(cfg)


    def forward(self, features, proposals, targets=None):
        """
        features: features from the backbone, can be multiple levels
        proposals: proposals from the RPN
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads

        x = self.feature_extractor(features, proposals)
        # x has shape (num_images * num_proposals_per_image)
        # print(x.shape)
        
        # final classifier that converts the features into predictions
        action_logits, target_mean = self.predictor(x)

        if not self.training:
            result = self.post_processor((action_logits, target_mean), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [action_logits], [target_mean]
        )
        
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

def build_roi_human_head(cfg, channels_in):
    return HumanCentricHead(cfg, channels_in)