import cv2 as cv
import numpy as np
import torch

path = 'dataset/'
img = cv.imread(path + 'IMG.JPG', cv.IMREAD_GRAYSCALE)
first_image = img.copy()

def display(img):
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww * (h / w))
    img = cv.resize(img, (neww, newh))
    cv.imshow('frameName', img)
    cv.waitKey(0)

kernel = np.ones((7, 7), np.uint8)
img = cv.GaussianBlur(img, (3, 3), 3)
img = cv.Canny(img, 70, 100)
img = cv.dilate(img, kernel, 5)

# display(img)
cont, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
squares = []
letters = []
for i in cont:
    x, y, w, h = cv.boundingRect(i)
    if w > 20 and h > 20:
        squares.append(cv.boundingRect(i))

for i in squares:
    x, y, w, h = i
    letters.append(cv.resize(first_image[y:y + h, x:x + w], (32, 32)))

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels=10, out_channels=10,kernel_size=3, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=10, out_channels = 30, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(6*6*30, 320),
            torch.nn.Tanh(),
            torch.nn.Linear(320, 160),
            torch.nn.Tanh(),
            torch.nn.Linear(160, 33))


    def forward(self, x):
        x = self.conv(x)
        return x

cnn = CNN()
cnn.load_set_