import cmapy
import cv2
import numpy as np
import torch


def visualize_return_layers(inputs, mid_outputs):
    final_img = inputs[0][0].detach().numpy()
    input_size = final_img.shape

    for value in mid_outputs.values():
        img = torch.mean(value, axis=1)[0].detach().numpy()
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_NEAREST)
        img = (img - np.min(img)) / np.max(img)

        final_img = cv2.hconcat([final_img, img])
    final_img = cv2.resize(
        final_img, (224 * len(mid_outputs) + 224, 224), interpolation=cv2.INTER_NEAREST
    )
    final_img = (final_img * 255).astype(np.uint8)
    im = cv2.applyColorMap(final_img, cmapy.cmap("viridis"))
    cv2.imshow("test", im)
    cv2.waitKey(3)
