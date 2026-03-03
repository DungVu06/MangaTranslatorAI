import torchvision
import yaml

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

def faster_rcnn(num_classes, anchor_sizes, anchor_ratios, box_nms_thresh=0.5):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, 
        box_nms_threshold=box_nms_thresh
    )

    formatted_sizes = tuple((size,) for size in anchor_sizes)
    formatted_ratios = (tuple(anchor_ratios),) * len(formatted_sizes)

    anchor_generator = AnchorGenerator(
        sizes=formatted_sizes,
        aspect_ratios=formatted_ratios
    )
    model.rpn.anchor_generator = anchor_generator

    out_channels = model.backbone.out_channels
    model.rpn.head = torchvision.models.detection.faster_rcnn.RPNHead(
        in_channels=out_channels, 
        num_anchors=anchor_generator.num_anchors_per_location()[0]
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

    return model

if __name__ == "__main__":
    yaml_mock_content = """
    model:
      num_classes: 3
      anchor_sizes: [32, 64, 128, 256, 512]
      anchor_ratios: [0.2, 0.5, 1.0, 2.0, 4.0]
    """

    config = yaml.safe_load(yaml_mock_content)
    model = faster_rcnn(
        num_classes=config["model"]["num_classes"],
        anchor_sizes=config["model"]["anchor_sizes"],
        anchor_ratios=config["model"]["anchor_ratios"],
        box_nms_thresh=config["model"]["box_nms_thresh"]
    )