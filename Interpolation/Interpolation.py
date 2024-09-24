import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr




def calculate_metrics(output, target):
    output_np = output
    target_np = target
    ssim_val = np.mean([ssim(o, t, data_range=t.max() - t.min(), channel_axis=-1, win_size=5) for o, t in zip(output_np, target_np)])
    psnr_val = np.mean([psnr(t, o, data_range=t.max() - t.min()) for o, t in zip(output_np, target_np)])
    mse_val = np.mean((output_np - target_np) ** 2)
    laplacian = cv2.Laplacian(output_np, cv2.CV_64F).var()
    return ssim_val, psnr_val, mse_val, laplacian


def main():

    sf = 4

    #input = cv2.imread('graffiti.png')
    input = cv2.imread('palmtree.png')
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    down_width = int(input.shape[:2][1] / sf)
    down_height = int(input.shape[:2][0] / sf)
    down_points = (down_width, down_height)
    points = (int(input.shape[:2][1]), int(input.shape[:2][0]))

    low_res = cv2.resize(input, down_points, interpolation=cv2.INTER_CUBIC)

    linear = cv2.resize(low_res, points, interpolation=cv2.INTER_LINEAR)
    nearest = cv2.resize(low_res, points, interpolation=cv2.INTER_NEAREST)
    lanczos = cv2.resize(low_res, points, interpolation=cv2.INTER_LANCZOS4)
    bicubic = cv2.resize(low_res, points, interpolation=cv2.INTER_CUBIC)

    # Display the input, low-res, and output images
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    laplacian = cv2.Laplacian(input, cv2.CV_64F).var()
    axs[0].imshow(input)
    axs[0].set_title(r"$\bf Original $" + f'\n\n\n\nClearness: {laplacian:.2f}')
    
    ssim_val, psnr_val, mse_val,laplacian = calculate_metrics(linear, input)
    axs[1].imshow(linear)
    axs[1].set_title(r"$\bf Linear $" +f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.3f}, \nMSE: {mse_val:.2f}, \nClarity: {laplacian:.1f}')
    ssim_val, psnr_val, mse_val, laplacian = calculate_metrics(nearest, input)
    axs[2].imshow(nearest)
    axs[2].set_title(r"$\bf Nearest $" + f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.3f}, \nMSE: {mse_val:.2f}, \nClarity: {laplacian:.1f}')

    ssim_val, psnr_val, mse_val, laplacian = calculate_metrics(bicubic, input)
    axs[3].imshow(bicubic)
    axs[3].set_title(r"$\bf Bicubic $" + f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.3f}, \nMSE: {mse_val:.2f}, \nClarity: {laplacian:.1f}')
    
    ssim_val, psnr_val, mse_val, laplacian = calculate_metrics(lanczos, input)
    axs[4].imshow(lanczos)
    axs[4].set_title(r"$\bf Lanczos $" + f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.3f}, \nMSE: {mse_val:.2f}, \nClarity: {laplacian:.1f}')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
