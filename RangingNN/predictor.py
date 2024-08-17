import threading
from pathlib import Path
import torch
import yaml

from RangingNN.YOLO1D import DetectionModel
from RangingNN.utils import get_cfg, Profile, select_device
from RangingNN.utils import DEFAULT_CFG, LOGGER
from RangingNN.model_utils import non_max_suppression, cw2lh


def normalize(spectrum):
    return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

class LoadTensor:
    """
    Load spectrums from torch.Tensor data.

    This class manages the loading and pre-processing of image data from PyTorch tensors for further processing.

    Attributes:
        im0 (torch.Tensor): The input tensor containing the spectrums(s).
        bs (int): Batch size, inferred from the shape of `im0`.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _single_check(im, stride): Validate and possibly modify the input tensor.
    """

    def __init__(self, im0) -> None:
        """Initialize Tensor Dataloader."""
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]

    @staticmethod
    def _single_check(im, stride=32):
        """Validate and format an image to torch.Tensor."""
        s = (
            f"WARNING ⚠️ torch.Tensor inputs should be BCH i.e. shape(batch, 1, 30720) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible.Added one extra channel dimention"
        )
        s2 = (
            f"WARNING ⚠️ The model was trained on spectrum with bin = 0.01 da and range 307.2 da"
            f"Input shape{tuple(im.shape)} is incompatible. Be aware of inaccurate results"
        )

        if len(im.shape) != 3:
            if len(im.shape) != 2:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im[:, None, :]

        if im.shape[2] % stride:
            raise ValueError(s)
        if im.shape[2] != 30720:
            LOGGER.warning(s2)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            LOGGER.warning(
                f"WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. "
                f"normalizing each spectrum"
            )
            im = [normalize(im_one.float()) for im_one in im]
            im = torch.stack(im)
        return im

    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    def __next__(self):
        """Return next item in the iterator."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.im0, [""] * self.bs

    def __len__(self):
        """Returns the batch size."""
        return self.bs

class DetectionPredictor:
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        modelPath (str): Model used for prediction.
        device (torch.device): Device used for prediction.
        inputTensor(Tensor): Data used for prediction.
        cfg(dict or path): Configuration dictionary for prediction.

    Example of use:
        import h5py, torch
        from RangingNN.predictor import DetectionPredictor
        with h5py.File(file, "r") as f:
            sp = torch.tensor(f['input'], dtype=torch.float32)[1:4]
        modelpath = "D:/APT_DATA/train_result/weights/best.pt"
        predictor = DetectionPredictor(modelpath, sp, save_dir = 'D:/APT_DATA/train_result', cfg = 'D:/APT_DATA/train_result/prediction_args.yaml')
        out = predictor()
    """

    def __init__(self, modelPath, inputTensor, save_dir, cfg=DEFAULT_CFG):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            the important args for prediction are conf: 0.25, iou: 0.7, max_det: 200, half: False
        """
        if isinstance(cfg, (str, Path)):
            cfg = yaml.safe_load(Path(cfg).read_text())

        self.args = get_cfg(cfg)
        self.device = select_device(self.args.device)
        self.save_dir = Path(save_dir)
        self.done_warmup = False
        self._lock = threading.Lock()  # for automatic thread-safe inference
        self.inputTensor = inputTensor
        self.modelPath = modelPath
        self.seen = 0
        self.speed = None

    def preprocess(self, sp):
        """Prepares input spectrum before inference."""
        return sp

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            multi_label=False,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def __call__(self, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.dataset = LoadTensor(self.inputTensor) # batch size is the first shape of the tensor
        return self.stream_inference()

    @torch.no_grad()
    def stream_inference(self, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("Start inference")

        # Setup model
        self.setup_model(self.modelPath)

        with self._lock:  # for thread-safe inference

            # Check if save_dir/ label file exists
            if self.args.save:
                self.save_dir.mkdir(parents=True, exist_ok=True)


            self.seen, self.batch = 0, None
            profilers = (Profile(device=self.device), Profile(device=self.device))
            for self.batch in self.dataset:
                im0s, s = self.batch

                # Inference
                with profilers[0]:
                    preds = self.model(im0s, *args, **kwargs)

                    # if self.args.embed:
                    #     yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                    #     continue
                #
                # Postprocess
                with profilers[1]:
                    self.results = self.postprocess(preds)

                # Visualize, save, write results
                n = len(im0s) # read first dimension
                for i in range(n):
                    self.seen += 1

                    # if self.args.verbose or self.args.save or self.args.show:
                    #     s[i] += self.write_results(i, Path(self.save_dir), im0s, s)

                # Print batch results
                self.speed = {
                    "inference": profilers[0].t * 1e3 / n,
                    "postprocess": profilers[1].t * 1e3 / n,

                }
                if self.args.verbose:
                    LOGGER.info(
                                "Speed:  %.1fms inference, %.1fms postprocess for per spectrum "
                                % tuple(self.speed.values())
                                )
                # yield from self.results

        # save final results
        if self.args.save:
            torch.save([re.to('cpu') for re in self.results], str(self.save_dir / "results_range_indexes.pt"))
            LOGGER.info(f"Results saved to {self.save_dir} \ results_range_indexes.pt ")
        return self.results

    def setup_model(self, modelpath):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        weights = torch.load(modelpath)
        self.model = DetectionModel()
        self.model.load(weights)
        self.model.nc = 1
        self.model.args = self.args  # attach hyperparameters to model
        self.model = self.model.to(self.args.device)

        self.model.device = self.device  # update device
        self.model.eval()

    # def write_results(self, i, im, s):
    #     """Write inference results to a file or directory."""
    #     string = ""  # print string
    #     if len(im.shape) == 2:
    #         im = im[None]  # expand for batch dim
    #
    #     string += f"{i}: "
    #     frame = self.dataset.count
    #     string += "%gx%g " % im.shape[2:]
    #     result = self.results[i]
    #     result.save_dir = self.save_dir.__str__()  # used in other locations
    #     string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
    #
    #     # Add predictions to image
    #     if self.args.save or self.args.show:
    #         self.plotted_img = result.plot(
    #             line_width=self.args.line_width,
    #             boxes=self.args.show_boxes,
    #             conf=self.args.show_conf,
    #             labels=self.args.show_labels,
    #             im_gpu=None if self.args.retina_masks else im[i],
    #         )
    #
    #     # Save results
    #     if self.args.show:
    #         self.show(self.save_dir)
    #     return string



