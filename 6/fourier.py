import numpy as np
import cv2

def ftransform(x, shift, inverse):
    row = x.shape[0]
    iterations = np.arange(row)
    indices = iterations.reshape((row, 1)) - (row // 2 if shift else 0)
    return np.dot(np.exp((2j if inverse else -2j) * np.pi * iterations * indices / row), x)

def ftransform2D(img, shift: bool = False, inverse: bool = False):
    divisors = img.shape if inverse else (1, 1)
    temp_transform = np.array([ftransform(y, shift, inverse) for y in img]) / divisors[1]
    return np.array([ftransform(x, shift, inverse) for x in temp_transform.T]).T / divisors[0]

def magnitude(array):
    return np.sqrt(array.real * array.real + array.imag * array.imag)

if __name__ == '__main__':
    threshold = 0.1
    img = cv2.imread('emma.jpeg', 0)

    transform = ftransform2D(img.astype(float), shift=True)
    mag = magnitude(np.log(transform))
    ft_magnitude = (mag * 255.0 / mag.max()).astype(np.uint8)

    _, mask = cv2.threshold(ft_magnitude, int(255 * threshold), 255, cv2.THRESH_BINARY)
    points = np.argwhere(mask).reshape((-1, 1, 2))

    lines = cv2.HoughLinesPointSet(points, 200, 10, min_rho=0, max_rho=300, max_theta=np.pi, theta_step=np.pi / 180, rho_step=1, min_theta=0)
    ft_inverse = magnitude(ftransform2D(transform, inverse=True))

    injection = 90 - np.degrees(lines[0][0][2])
    im = img.copy()
    rows, columns = im.shape
    rotate = cv2.warpAffine(im, cv2.getRotationMatrix2D((rows // 2, columns // 2), injection, 1.0), (rows, columns))
    cv2.imshow('FT', ft_inverse.astype(np.uint8))
    cv2.imshow('Magnitude', ft_magnitude)
    # cv2.imshow('Rotation', rotate)
    cv2.waitKey(0) & 0xFF == ord('q')
