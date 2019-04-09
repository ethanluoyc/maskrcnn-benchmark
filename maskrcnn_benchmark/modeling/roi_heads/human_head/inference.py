# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .head2 import split_person_object, encode_object_location, compatibility


def _score(humaness, objectness, actioness, compatibility):
    # returns score <num_humans, num_objects, num_actions>
    # TODO make actioness object specific.
    humaness = humaness.view(-1, 1, 1) # nh x 1 x 1
    objectness = objectness.view(1, -1, 1) # # 1 x no x 1
    actioness = actioness.unsqueeze(1) # nh x 1 x na    
    return humaness * objectness * actioness * compatibility # nh x no x na


def score_bbox(person_box, object_box,
               num_actions, 
               person_ind=2):
    num_humans = len(person_box.bbox)
    num_objects = len(object_box.bbox)

    humaness = person_box.get_field('scores')[:, person_ind]
    object_score, _ = object_box.get_field('scores').max(dim=1)

    # TODO compute these rather than hardcode
    actioness, target_mean = MockHumanBranch(person_box, num_actions)

    encoded_object = encode_object_location(object_box.bbox, person_box.bbox)
    compat = compatibility(encoded_object, target_mean, std=0.3)

    return _score(humaness, object_score, actioness, compat)



class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        std=0.3,
        person_idx=2
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.std = std
        self.person_idx = person_idx

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the action logits, target mean
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        assert len(boxes) == 1, "Non-batched for now"

        action_logits, target_mean = x
        class_prob = F.softmax(action_logits, -1)

        # # TODO think about a representation of batch of boxes
        # image_shapes = [box.size for box in boxes]
        # boxes_per_image = [len(box) for box in boxes]
        # concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # proposals = self.box_coder.decode(
        #     target_mean.view(sum(boxes_per_image), -1), concat_boxes
        # )

        num_actions = class_prob.shape[1]

        results = []
        for box in boxes:
            scores = box.get_field('scores')
            best_scores, best_score_idx = scores.max(dim=1)

            person_mask = best_score_idx == self.person_idx
            object_mask = ~person_mask
            
            persons = box.bbox[person_mask]
            objects = box.bbox[object_mask]

            num_humans = len(persons)
            num_objects = len(objects)

            humaness = best_scores[person_mask] # (num_humans, )
            object_score = best_scores[object_mask] # (num_objects )

            actioness = action_logits[person_mask].reshape(num_humans, num_actions)
            target_mean = target_mean[person_mask].reshape(num_humans, num_actions, 4)

            encoded_object = encode_object_location(objects, persons)

            compat = compatibility(encoded_object, target_mean, std=self.std)

            # nh x no x na
            scores = _score(humaness, object_score, actioness, compat)

            results.append(scores)


        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_human_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    std = cfg.MODEL.ROI_HEADS.STANDARD_DEVIATION

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        std
    )
    return postprocessor
