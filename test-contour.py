import cv2
import numpy as np

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def set_image(image, colormap):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_new = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    for idx,i in enumerate(colormap):
        image_new[:, :, 2-idx] = i
    image_new[gray == 0] = 0

    _, thresh = cv2.threshold(gray, 0, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_new, contours, -1, (0, 0, 255), 3)
    return image_new


img = cv2.imread('cake.png')
cmp = color_map(5)
new_img = set_image(img, cmp[1])
cv2.imshow("image", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()