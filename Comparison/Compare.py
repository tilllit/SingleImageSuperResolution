import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import correlation

from TF import *
from GAN import *




def calculate_metrics(output, target):
    output_np = output
    target_np = target
    ssim_val = np.mean([ssim(o, t, data_range=t.max() - t.min(), channel_axis=-1, win_size=5) for o, t in zip(output_np, target_np)])
    psnr_val = np.mean([psnr(t, o, data_range=t.max() - t.min()) for o, t in zip(output_np, target_np)])
    scc_val = correlation(output_np.flatten(), target_np.flatten(), centered=True)
    #print("scc", scc_val)
    laplacian = cv2.Laplacian(output_np, cv2.CV_64F).var()
    #print("laplacian", laplacian)
    mse_val = np.mean((output_np - target_np) ** 2)
    return ssim_val, psnr_val, mse_val, scc_val, laplacian


def main():

    original_path = '_test/org.png'
    cnn_path = '_test/pred/cnn.png'
    srgan_path = '_test/pred/gan.png'

    test_cnn(original_path)
    test_gan(original_path)

    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)


    sf = 4
    down_width = int(original.shape[:2][1] / sf)
    down_height = int(original.shape[:2][0] / sf)
    down_points = (down_width, down_height)
    resized_down = cv2.resize(original, down_points, interpolation=cv2.INTER_CUBIC)
    lanczos = cv2.resize(np.asarray(resized_down), (int(original.shape[:2][1]), int(original.shape[:2][0])), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("_test/pred/lanczos.png", cv2.cvtColor(lanczos, cv2.COLOR_BGR2RGB))

    cnn = cv2.imread(cnn_path)
    cnn = cv2.cvtColor(cnn, cv2.COLOR_BGR2RGB)

    srgan = cv2.imread(srgan_path)
    srgan = cv2.cvtColor(srgan, cv2.COLOR_BGR2RGB)

    # Display the input, low-res, and output images
    fig = plt.figure("Compare 4x", figsize=(20,5))
    axs = fig.subplots(1, 4, sharex=True, sharey=True)
    plt.tight_layout()


    laplacian = cv2.Laplacian(original, cv2.CV_64F).var()
    axs[0].imshow(original)
    axs[0].set_title(r"$\bf Original $" + f'\n\n\n\nLapl: {laplacian:.2f}')

    # zm = axs[0].inset_axes([0.7,0.7,0.3,0.3])
    # zm.imshow(original[800:900,1700:1800])
    # axs[0].indicate_inset_zoom(zm, lw=10)
    
    ssim_val, psnr_val, mse_val, scc_val, laplacian = calculate_metrics(lanczos, original)
    axs[1].imshow(lanczos)
    axs[1].set_title(r"$\bf Lanczos $" + f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.4f}, \nMSE: {mse_val:.6f}, \nLapl: {laplacian:.2f}')
    
    ssim_val, psnr_val, mse_val, scc_val, laplacian = calculate_metrics(cnn, original)
    axs[2].imshow(cnn)
    axs[2].set_title(r"$\bf CNN $" + f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.4f}, \nMSE: {mse_val:.6f}, \nLapl: {laplacian:.2f}')

    ssim_val, psnr_val, mse_val, scc_val, laplacian = calculate_metrics(srgan, original)
    axs[3].imshow(srgan)
    axs[3].set_title(r"$\bf SRGAN $" + f'\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.4f}, \nMSE: {mse_val:.6f}, \nLapl: {laplacian:.2f}')

    plt.show()


if __name__ == '__main__':
    main()
