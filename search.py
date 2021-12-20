import cv2 as cv
import numpy as np
import torch

path = 'dataset/'
img = cv.imread(path + 'IMG.JPG', cv.IMREAD_GRAYSCALE)
rot = cv.getRotationMatrix2D((img.shape[0]//2, img.shape[1]//2), 90, 1)
img = cv.warpAffine(img, rot, (img.shape[0], img.shape[1]))
first_image = img.copy()


def display(img):
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww * (h / w))
    img = cv.resize(img, (neww, newh))
    cv.imshow('frameName', img)
    cv.waitKey(0)

# display(img)


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
    if w > 20 and h > 20 and w < 400 and h < 400:
        squares.append(cv.boundingRect(i))

# a = first_image
# for i in squares:
#     a = cv.rectangle(a, i, (255, 0, 0), 5)
# display(a)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=1, padding=0),
            torch.nn.Conv2d(10, 20, 7, padding=2),
            torch.nn.BatchNorm2d(20),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=6),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),

            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(9*9*40, 700),
            torch.nn.Tanh(),
            torch.nn.Linear(700, 300),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(300, 33),
            torch.nn.Softmax(1)
            )


    def forward(self, x):
        self.train()
        x = self.conv(x)
        return x

    def test(self, x):
        self.eval()
        return self.conv[0:-1](x)

    def pred(self, x):
        self.eval()
        return self.conv(x)



cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pkl'))

classes = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е',\
           'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

def pred(img):
    img = cv.resize(img, (48, 48))
    img = img.reshape(1, 1, 48, 48)
    img = torch.from_numpy(img).float()
    return classes[cnn.pred(img).argmax()]
# print(squares[0])
# x, y, w, h = squares[0]
# cv.imshow('a', img[y:y + h, x:x + w])
# cv.waitKey(0)



img = cv.bitwise_not(img)
for idx, i in enumerate(squares):
    x, y, w, h = i
    crop = cv.resize(img[y:y + h, x:x + w], (48, 48))

    letters.append(crop)
    # print(idx, ':', pred(crop))
    img = cv.rectangle(img, i, 0, 4)
    img = cv.putText(img, idx.__str__(), (x, y), cv.FONT_HERSHEY_SIMPLEX, 5, 0, 5)
cnn.eval()
nt = torch.tensor(np.array(letters)).reshape(7, 1, 48, 48).float()
print(nt.shape)
for idx, i in enumerate(cnn.pred(nt).argmax(1)):
    print(idx, ':', classes[i])

display(img)


# print(pred(letters[0]))
# print(len(letters))
# print(letters)
# print(pred(letters[2]))
# cv.imshow('a', letters[2])
# cv.waitKey(0)



# # display(img)
# print(len(letters))
# cv.imshow('1', letters[0])
# cv.waitKey(0)