import torch
from peak_detection.RangingNN.modules import *
from copy import deepcopy
from pathlib import Path
import contextlib
from peak_detection.RangingNN.loss import v8DetectionLoss
from peak_detection.RangingNN.utils import yaml_load
from peak_detection.RangingNN.utils import LOGGER


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv1d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm1d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


# https://github.com/ultralytics/ultralytics/blob/b3dfbdda0239037e83b9949ebd4fb9758a75032e/ultralytics/nn/tasks.py#L830
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            # LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{'activation:'} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            ConvTranspose,
            DWConv,
            C2f,
            SPPF,
        }:
            c1, c2 = ch[f], args[0]
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in {C2f, }:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm1d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        # if verbose:
        #     LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8.yaml", ch=1, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)  # cfg dict
        # self.yaml = cfg ############ Let me just figure out the dict and use the dict
        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        # if nc and nc != self.yaml["nc"]:
        #     LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        #     self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            # (1, ch, s, s) changed to (1, ch, s)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        # if verbose:
        #     self.info()
        #     LOGGER.info("")

    # def _predict_augment(self, x):
    #     """Perform augmentations on input image x and return augmented inference and train outputs."""
    #     img_size = x.shape[-2:]  # height, width
    #     s = [1, 0.83, 0.67]  # scales
    #     f = [None, 3, None]  # flips (2-ud, 3-lr)
    #     y = []  # outputs
    #     for si, fi in zip(s, f):
    #         xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #         yi = super().predict(xi)[0]  # forward
    #         yi = self._descale_pred(yi, fi, si, img_size)
    #         y.append(yi)
    #     y = self._clip_augmented(y)  # clip augmented tails
    #     return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)
