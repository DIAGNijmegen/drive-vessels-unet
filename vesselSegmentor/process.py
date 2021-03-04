
import SimpleITK
import numpy as np
import torch
import monai
from scipy.special import expit
from skimage import transform

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Vesselsegmentor(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load("../best_metric_model_segmentation2d_dict.pth"))

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:

        image = SimpleITK.GetArrayFromImage(input_image)
        image = np.array(image)
        image = transform.resize(image, (512, 512), order=3)
        image = image.astype(np.float32) / 255.
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).reshape(1, 3, 512, 512)
        out = self.model(image).squeeze().data.cpu().numpy()
        out = (expit(out) > 0.99).astype(np.uint8)
        out = SimpleITK.GetImageFromArray(out)
        return out


if __name__ == "__main__":
    Vesselsegmentor().process()