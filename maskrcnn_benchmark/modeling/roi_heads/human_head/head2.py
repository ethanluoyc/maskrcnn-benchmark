import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

def score(humaness, objectness, actioness, compatibility):
    # returns score <num_humans, num_objects, num_actions>
    # humaness (num_humans, )
    # objectness (num_objects)
    # actionness torch.randn(num_humans, num_actions)
    # compatibility (num_humans, num_objects, num_actions)
    # TODO make actioness object specific.
    humaness = humaness.view(-1, 1, 1) # nh x 1 x 1
    objectness = objectness.view(1, -1, 1) # # 1 x no x 1
    actioness = actioness.unsqueeze(1) # nh x 1 x na    
    return humaness * objectness * actioness * compatibility

def encode_object_location(object_box, human_box):

    """
    object_box (m x 4)
    human_box (n x 4)
    
    N.B. assuming xywh encoding
    # TODO box coder instead?

    Returns
    b_{o|h}: (n x m x 4)
    """

    xo = object_box[:, 0].unsqueeze(0) # 1 x m
    yo = object_box[:, 1].unsqueeze(0) # 1 x m
    wo = object_box[:, 2].unsqueeze(0) # 1 x m
    ho = object_box[:, 3].unsqueeze(0) # 1 x m


    xh = human_box[:, 0].unsqueeze(-1) # n x 1
    yh = human_box[:, 1].unsqueeze(-1) # ..
    wh = human_box[:, 2].unsqueeze(-1) # ..
    hh = human_box[:, 3].unsqueeze(-1) # ..

    return torch.stack([
        (xo - xh).div(wh),
        (yo - yh).div(hh),
        wo.log() - wh.log(), # log(x/y) = logx - y for numerical stability
        ho.log() - hh.log()
    ]).permute(1, 2, 0)    

def compatibility(object_box_enc, target_pred, std=0.3):
    """
    object_box_enc: (n x m x 4)
    target_pred: (n x num_actions x 4)
    
    Returns
    g^a_{h,o}: (num_humans, num_objects, num_actions)
    """
    target_pred = target_pred.unsqueeze(1) # n x 1 x num_actions x 4
    object_box_enc = object_box_enc.unsqueeze(2) # n x m x 1 x 4
    
    distance = (object_box_enc - target_pred).pow(2).mean(-1)
    return torch.exp(-distance.div(2*std*std))

def split_person_object(bbox, person_idx):
    scores = bbox.get_field('scores')
    best_score_idx = scores.argmax(dim=1)
    person_mask = best_score_idx == person_idx
    object_mask = ~(best_score_idx == person_idx)
    
    persons = BoxList(bbox.bbox[person_mask], bbox.size, bbox.mode)
    objects = BoxList(bbox.bbox[object_mask], bbox.size, bbox.mode)
    
    for field in bbox.fields():
        persons.add_field(field, bbox.get_field(field)[person_mask])
        objects.add_field(field, bbox.get_field(field)[object_mask])
    
    return persons, objects

def MockHumanBranch(human_box, num_actions):
    num_humans = len(human_box)
    return torch.randn(num_humans, num_actions).abs(), torch.randn(num_humans, num_actions, 4).abs()


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

    return score(humaness, object_score, actioness, compat)

if __name__ == "__main__":
    num_humans = 2
    num_objects = 10
    num_actions = 8
    num_classes = 20

    bbox = torch.randn(3, 4).abs()
    box = BoxList(bbox, image_size=(20, 20), mode='xywh')
    scores = torch.randn(3, 3)
    person_ind = 2
    box.add_field('scores', torch.tensor([[1,2,3],[4,6,5], [9,8,7]], dtype=torch.float32))
    person_box, object_box = split_person_object(box, person_ind)

    score_bbox(person_box, object_box, num_actions=10)