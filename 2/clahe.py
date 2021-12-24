import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def interpolate(bin, LU, RU, LB, RB, X, Y):
    subIm = np.zeros(bin.shape)
    num = X * Y
    for col in range(X):
        invX = X - col
        for row in range(Y):
            invY = Y - row
            val = bin[col, row].astype(int)
            subIm[col, row] = np.floor(
                (invX * (invY * LU[val] + row * RU[val]) + col * (invY * LB[val] + row * RB[val])) / float(num))
    return subIm

def makeHist(img, clipLimit, Bins=128, x=0, y=0):
    row, column = img.shape

    Bins = max(Bins, 128)
    if x == 0:
        xsz = 32
        ysz = 32
        x = np.ceil(row / xsz).astype(int)  # 240
        excX = int(xsz * (x - row / xsz))
        y = np.ceil(column / ysz).astype(int)  # 320
        excY = int(ysz * (y - column / ysz))
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)
    else:
        xsz = round(row / x)
        ysz = round(column / y)

    minVal = 0
    maxVal = 255

    binSz = np.floor(1 + (maxVal - minVal) / float(Bins))
    LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / float(binSz))

    bins = LUT[img]
    hist = np.zeros((x, y, Bins))
    print(x, y, hist.shape)
    for i in range(x):
        for j in range(y):
            bin_ = bins[i * xsz:(i + 1) * xsz, j * ysz:(j + 1) * ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i, j, bin_[i1, j1]] += 1



if __name__ == '__main__':
    image = io.imread('emma.jpeg')
    hist = makeHist(image[:, :, 0], 8, 0, 0)

    fig, axs = plt.subplots(1, 2, figsize=(200, 100))
    axs[0].imshow(image[:, :, 0], cmap='gray')
    plt.show()