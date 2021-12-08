from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import imageio

def Interpolation(val, Grad, Gx, Gy):
    supression = np.zeros(val.shape)

    for i in range(1, int(val.shape[0]) - 1):
        for j in range(1, int(val.shape[1]) - 1):
            if ((Grad[i, j] > 45 and Grad[i, j] <= 90) or (Grad[i, j] < -90 and Grad[i, j] >= -135)):
                yUp = np.array([val[i + 1, j], val[i + 1, j + 1]])
                ydown = np.array([val[i - 1, j], val[i - 1, j - 1]])
                x_est = np.absolute(Gx[i, j] / val[i, j])
                if (val[i, j] >= ((yUp[1] - yUp[0]) * x_est + yUp[0]) and val[i, j] >= (
                        (ydown[1] - ydown[0]) * x_est + ydown[0])):
                    supression[i, j] = val[i, j]
                else:
                    supression[i, j] = 0

            if ((Grad[i, j] > 135 and Grad[i, j] <= 180) or (Grad[i, j] < 0 and Grad[i, j] >= -45)):
                yUp = np.array([val[i, j - 1], val[i + 1, j - 1]])
                ydown = np.array([val[i, j + 1], val[i - 1, j + 1]])
                x_est = np.absolute(Gy[i, j] / val[i, j])
                if (val[i, j] >= ((yUp[1] - yUp[0]) * x_est + yUp[0]) and val[i, j] >= (
                        (ydown[1] - ydown[0]) * x_est + ydown[0])):
                    supression[i, j] = val[i, j]
                else:
                    supression[i, j] = 0

            if ((Grad[i, j] > 90 and Grad[i, j] <= 135) or (Grad[i, j] < -45 and Grad[i, j] >= -90)):
                yUp = np.array([val[i + 1, j], val[i + 1, j - 1]])
                ydown = np.array([val[i - 1, j], val[i - 1, j + 1]])
                x_est = np.absolute(Gx[i, j] / val[i, j])
                if (val[i, j] >= ((yUp[1] - yUp[0]) * x_est + yUp[0]) and val[i, j] >= (
                        (ydown[1] - ydown[0]) * x_est + ydown[0])):
                    supression[i, j] = val[i, j]
                else:
                    supression[i, j] = 0

            if ((Grad[i, j] >= 0 and Grad[i, j] <= 45) or (Grad[i, j] < -135 and Grad[i, j] >= -180)):
                yUp = np.array([val[i, j + 1], val[i + 1, j + 1]])
                ydown = np.array([val[i, j - 1], val[i - 1, j - 1]])
                x_est = np.absolute(Gy[i, j] / val[i, j])
                if (val[i, j] >= ((yUp[1] - yUp[0]) * x_est + yUp[0]) and val[i, j] >= (
                        (ydown[1] - ydown[0]) * x_est + ydown[0])):
                    supression[i, j] = val[i, j]
                else:
                    supression[i, j] = 0

    return supression

def HystTresh(img):
    hyst = np.copy(img)
    h = int(hyst.shape[0])
    w = int(hyst.shape[1])
    maxthreshrat = 0.2
    lowthreshrat = 0.15
    maxthresh = np.max(hyst) * maxthreshrat
    lowthresh =  maxthresh * lowthreshrat
    x = 0.1
    pred = 0.0

    while (pred != x):
        pred = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (hyst[i, j] >  maxthresh):
                    hyst[i, j] = 1
                elif (hyst[i, j] < lowthresh):
                    if (
                            (hyst[i - 1, j + 1] > maxthresh) or
                            (hyst[i - 1, j - 1] > maxthresh) or
                            (hyst[i + 1, j - 1] > maxthresh) or
                            (hyst[i, j - 1] > maxthresh) or
                            (hyst[i, j + 1] > maxthresh) or
                            (hyst[i - 1, j] > maxthresh) or
                            (hyst[i + 1, j] > maxthresh) or
                            (hyst[i + 1, j + 1] > maxthresh)):
                        hyst[i, j] = 1
                    hyst[i, j] = 0
                else:
                    pass
        x = np.sum(hyst == 1)

    hyst = (hyst == 1) * hyst

    return hyst

if __name__ == "__main__":
    img = imageio.imread("emma.jpeg")
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    blur = ndimage.gaussian_filter(gray, sigma=1.4)

    dx = ndimage.sobel(blur, axis=1, mode='constant', cval=0.0)
    dy = ndimage.sobel(blur, axis=0, mode='constant', cval=0.0)

    value = np.hypot(dx, dy)
    value = value / np.max(value)

    gradient = np.degrees(np.arctan2(dy, dx))

    nms = Interpolation(value, gradient, dx, dy)
    nms = nms / np.max(nms)

    Final_Image = HystTresh(nms)
    plt.imshow(Final_Image, cmap=plt.get_cmap('gray'))
    plt.show()