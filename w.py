import cv2 as cv
import torch
import numpy as np

# This is test...
# img = cv.imread('M.jpeg', cv.IMREAD_GRAYSCALE)
# img = cv.resize(img, (32, 32))
# rot = cv.getRotationMatrix2D((16, 16), 270, 1)
# img = cv.warpAffine(img, rot, (img.shape[0], img.shape[1]))


classes = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
           'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=10, out_channels=30, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(6 * 6 * 30, 320),
            torch.nn.Tanh(),
            torch.nn.Linear(320, 160),
            torch.nn.Tanh(),
            torch.nn.Linear(160, 33))

    def forward(self, x):
        x = self.conv(x)
        return x


cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pkl'))

path = 'dataset/'
img = cv.imread(path + 'IMG.JPG', cv.IMREAD_GRAYSCALE)
first_image = img.copy()

kernel = np.ones((7, 7), np.uint8)
img = cv.GaussianBlur(img, (3, 3), 3)
img = cv.Canny(img, 70, 100)
img = cv.dilate(img, kernel, 5)


cont, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
squares = []
letters = []

for i in cont:
    x, y, w, h = cv.boundingRect(i)
    if w > 20 and h > 20:
        squares.append(cv.boundingRect(i))

def pred(img):
    new = img.copy()
    new = cv.resize(new, (32, 32))
    new = torch.tensor(new).reshape(1, 1, 32, 32)
    new = new.float()
    new /= 255
    pr = cnn.forward(new)
    value = classes[pr.argmax()]
    return value

img = cv.bitwise_not(img)
for idx, i in enumerate(squares):
    x, y, w, h = i
    crop = cv.resize(img[y:y + h, x:x + w], (32, 32))
    letters.append(crop)
    img = cv.rectangle(img, (x, y, w, h), (0), 10)
    img = cv.putText(img, idx.__str__(), (x+w+2, y), cv.FONT_HERSHEY_SIMPLEX, 2, (0), 2)

# cv.imshow('let', letters[0])
# cv.waitKey(0)
# input only one image to view numpy.array




# print(pred(letters[1]))
# cv.imshow('let', letters[1])
# cv.waitKey(0)


for idx, i in enumerate(letters):
    print('num', idx, pred(i))

cv.imshow('value', cv.resize(img, (800, 1200)))
cv.waitKey(0)

# m = cv.imread('E.jpeg', cv.IMREAD_GRAYSCALE)
# rot = cv.getRotationMatrix2D((139, 139), 90, 1)
# m = cv.warpAffine(m, rot, (m.shape[0], m.shape[1]))
# print(pred(m))
# cv.imshow('m', m)
# cv.waitKey(0)