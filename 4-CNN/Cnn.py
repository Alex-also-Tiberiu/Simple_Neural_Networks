import numpy as np
import torchvision

root_dir = './'
torchvision.datasets.MNIST(root=root_dir,download=True)

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image):
      '''
      Generates all possible 3x3 image regions using valid padding.
      - image is a 2d numpy array
      '''
      h, w = image.shape

      for i in range(h - 2):
          for j in range(w - 2):
              im_region = image[i:(i + 3), j:(j + 3)]
              yield im_region, i, j

  def forward(self, input):
      '''
      Performs a forward pass of the conv layer using the given input.
      Returns a 3d numpy array with dimensions (h, w, num_filters).
      - input is a 2d numpy array
      '''
      h, w = input.shape
      output = np.zeros((h - 2, w - 2, self.num_filters))

      for im_region, i, j in self.iterate_regions(input):
          # Convolution
          output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

      return output


class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      # Pooling operation taking the max value
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output


class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    input = input.flatten()

    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases
    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)


# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)

test_images = torchvision.datasets.MNIST(root=root_dir, train=False, download=True)
test_labels = test_images.targets

conv = Conv3x3(8)
pool = MaxPool2()
output = conv.forward(np.array(train_images[0][0]))
print(output.shape) # (26, 26, 8)
output = pool.forward(output)
print(output.shape) # (13, 13, 8)

softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
  # Do a forward pass.
  image_data = np.array(im[0])
  o, l, acc = forward(image_data, label)
  loss += l
  num_correct += acc

  # Print stats every 100 steps.
  if i % 100 == 99:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, num_correct)
    )
    loss = 0
    num_correct = 0