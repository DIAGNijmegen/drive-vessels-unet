from PIL import Image
import torch
import monai
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = monai.networks.nets.UNet(
    dimensions=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load("./best_metric_model_segmentation2d_dict.pth"))

image = Image.open("D:/data/DRIVE/test/images/01_test.tif")
image = image.resize((512, 512), resample=2)
image = np.array(image)
