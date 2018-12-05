import os, glob, cv2
import numpy as np
from sklearn.model_selection import train_test_split

if not os.path.isdir("four_shapes"):
    os.mkdir("four_shapes")

circles = list(sorted(glob.glob(os.path.join('circle/', '*.png'))))
triangles = list(sorted(glob.glob(os.path.join('triangle/', '*.png'))))
squares = list(sorted(glob.glob(os.path.join('square/', '*.png'))))
stars = list(sorted(glob.glob(os.path.join('star/', '*.png'))))
shapes = [circles, triangles, squares, stars]

img_inputs = []
for shape in shapes:
    for f in shape:
        img = cv2.imread(f, 0)
        img = img.astype(float)/255.
        img = img.reshape((img.shape[0], img.shape[1], 1))
        img_inputs.append(img)

img_inputs = np.array(img_inputs)
img_train, img_test = train_test_split(img_inputs, test_size=0.2)

print('train shape', img_train.shape)
print('test shape', img_test.shape)

np.save('four_shapes/train_shapes.npy', img_train)
np.save('four_shapes/test_shapes.npy', img_test)

fake_ytrain = np.zeros(shape=img_train.shape[0], dtype=int)
fake_ytest = np.zeros(shape=img_test.shape[0], dtype=int)

print('y train shape', fake_ytrain.shape)
print('y test shape', fake_ytest.shape)

np.save('four_shapes/train_labels.npy', fake_ytrain)
np.save('four_shapes/test_labels.npy', fake_ytest)
