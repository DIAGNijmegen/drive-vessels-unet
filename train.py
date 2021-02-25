import logging
import os
import sys
from glob import glob
import numpy as np

import torch
import torch.nn.functional as F
from dataloader import DRIVEDataset, Rescale, ToTensor, Normalize, WeightMap
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.visualize import plot_2d_or_3d_image


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    IMAGE_ROOT = "D:/data/DRIVE/training/images"
    LABEL_ROOT = "D:/data/DRIVE/training/1st_manual"

    images = glob(os.path.join(IMAGE_ROOT, "*training.tif"))
    labels = glob(os.path.join(LABEL_ROOT, "*manual1.gif"))

    data_transform = transforms.Compose([
        Rescale((512, 512)),
        Normalize(),
        WeightMap(),
        ToTensor(),
    ])

    train_ds = DRIVEDataset(images[:-10], labels[:-10], transform=data_transform)
    valid_ds = DRIVEDataset(images[-10:], labels[-10:], transform=data_transform)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True,
    )

    val_loader = DataLoader(
        valid_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True,
    )

    # create UNet and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

    # start a typical PyTorch training
    epochs_total = 1000
    val_interval = 1
    best_loss = np.inf
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(epochs_total):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs_total}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels, weights = (
                batch_data["img"].to(device),
                batch_data["seg"].to(device),
                batch_data["map"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                epoch_val_loss = []
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels, val_weights = (
                        val_data["img"].to(device),
                        val_data["seg"].to(device),
                        val_data["map"].to(device),
                    )
                    val_outputs = model(val_images).squeeze()
                    loss = loss_function(val_outputs, val_labels)
                    epoch_val_loss.append(loss.item())
                epoch_val_loss = np.array(epoch_val_loss).mean()
                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(), "best_metric_model_segmentation2d_dict.pth"
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current val loss: {:.4f} best val loss: {:.4f} at epoch {}".format(
                        epoch + 1, epoch_val_loss, best_loss, best_metric_epoch
                    )
                )
                writer.add_scalar("val_loss", epoch_val_loss, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs, epoch + 1, writer, index=0, tag="output"
                )

    print(f"train completed, best_loss: {best_loss:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":

    main()
