import torch
import time
import numpy as np
from RangingNN.utils import LOGGER
# https://github.com/ultralytics/ultralytics/blob/8d17af7e32ac3b536bdcced7b7705ce688ae6d94/ultralytics/utils/ops.py#L162


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, w = feats[i].shape
        sf = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift
        anchor_points.append(sf.view(-1, 1))  # shape [long, 1]
        stride_tensor.append(torch.full((w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2bbox(distance, anchor_points, cw=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    l, h = distance.chunk(2, dim)
    x1y1 = anchor_points - l
    x2y2 = anchor_points + h
    if cw:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def cw2lh(x):
    """
    Convert bounding range coordinates from (center, width) format to (low, high) format
    Args:
        x (np.ndarray | torch.Tensor)

    Returns:
        y (np.ndarray | torch.Tensor)
    """
    assert x.shape[-1] == 2, f"input shape last dimension expected 2 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = x[..., 0] - x[..., 1] / 2.0
    y[..., 1] = x[..., 0] + x[..., 1] / 2.0

    return y


def lh2cw(x):
    """
    Convert bounding range coordinates from (low, high) format to (center, width) format
    Args:
        x (np.ndarray | torch.Tensor)

    Returns:
        y (np.ndarray | torch.Tensor)
    """
    assert x.shape[-1] == 2, f"input shape last dimension expected 2 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 1]) / 2.0
    y[..., 1] = x[..., 1] - x[..., 0]

    return y


def scale_boxes(ranges, ratio_pad, spectrumsz):
    """
    Rescales bounding ranges (in the format of lh by default) from img1_shape to the shape
    of a different image (img0_shape).

    Args:
        ranges: the orignal ranges loaded from the label.
        spectrumsz: the shape of resized spectrum
        ratio_pad (float): a number of ratio for scaling the boxes. Pad is not concidered for now.

    Returns:
        The scaled bounding ranges, in the format of (low, high)
    """

    ranges[..., :] /= ratio_pad[0] # TODO feels like it should be *
    if isinstance(ranges, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        ranges[..., 0] = ranges[..., 0].clamp(0, spectrumsz)  # low
        ranges[..., 1] = ranges[..., 1].clamp(0, spectrumsz)  # high
    else:
        ranges[..., :] = ranges[..., :].clip(0, spectrumsz)
        # np.array (faster grouped)
    return ranges


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=100,
        in_place=True,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 2 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 2)  # number of classes
    nm = prediction.shape[1] - nc - 2
    mi = 2 + nc  # mask start index
    xc = prediction[:, 2:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if in_place:
        prediction[..., :2] = cw2lh(prediction[..., :2])  # xywh to xyxy
    else:
        prediction = torch.cat((cw2lh(prediction[..., :2]), prediction[..., 2:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # # Cat apriori labels if autolabelling
        # if labels and len(labels[xi]) and not rotated:
        #     lb = labels[xi]
        #     v = torch.zeros((len(lb), nc + nm + 2), device=x.device)
        #     v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
        #     v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
        #     x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((2, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 3:4] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 2].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 3:4] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 2]  # scores
        boxes = x[:, :2] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output
