import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import SimpleITK
import torch
import monai
import numpy as np
from skimage import transform
from scipy.special import expit


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

image = SimpleITK.ReadImage("vesselSegmentor/test/1000.0.tif")
image = SimpleITK.GetArrayFromImage(image)
image = np.array(image)
shape = image.shape
image = transform.resize(image, (512, 512), order=3)
image = image.astype(np.float32) / 255.
image = image.transpose((2, 0, 1))
image = torch.from_numpy(image).to(device).reshape(1, 3, 512, 512)
out = model(image).squeeze().data.cpu().numpy()
out = transform.resize(out, shape[:-1], order=3)
out = (expit(out) > 0.99).astype(np.uint8)
out = SimpleITK.GetImageFromArray(out)
SimpleITK.WriteImage(out, "vesselSegmentor/test/1000.1.mha", True)