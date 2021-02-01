# Train-your-custom-Object-Detection-Model-with-CNN

## IMPORT NECESSARY MODULES ##
```
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
from skimage.io import imread
from google.colab import drive
import os
import sys
import random
import math
import re
import time

import matplotlib

import xml.etree.ElementTree as ET
 
drive.mount('/gdrive')
```
## CREATE A NEW DATASET CLASS ##
```
class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids
   ```
   ## Create a new Custom Dataset Class with the Class Lables ##
   ```
   
class Additive_Manu_Dataset(Dataset):
    def load_dataset(self, dataset_dir,is_train=True):
        dataset_dir= '/gdrive/MyDrive/tensorflow1/additive_manufacturing/data/train_labels'
        if is_train==False:
         dataset_dir='/gdrive/MyDrive/tensorflow1/additive_manufacturing/data/test_labels'
        self.add_class('dataset', 1, 'Layer 1')
        self.add_class('dataset', 2, 'Layer 2')
        self.add_class('dataset',3, 'Layer 3')
        self.add_class('dataset',4, 'Layer 4')
        self.add_class('dataset',5, 'Layer 5')
        self.add_class('dataset',6, 'Layer 6')
        self.add_class('dataset',7, 'Layer 7')
        self.add_class('dataset',8, 'Layer 8')
        self.add_class('dataset',9, 'Layer 9')
        self.add_class('dataset',10, 'Layer 10')
        self.add_class('dataset',11, 'Layer 11')
        self.add_class('dataset',12, 'defect')
        
        
        # find all images
        for i, filename in enumerate(os.listdir(dataset_dir)):
            if '.jpg' in filename:
                self.add_image('dataset', 
                               image_id=i, 
                               path=os.path.join(dataset_dir, filename), 
                               annotation=os.path.join(dataset_dir, filename.replace('.jpg', '.xml')))
    
    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ET.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = []
        classes = []
        for member in root.findall('object'):
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(self.class_names.index(member[0].text))
        # extract image dimensions
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        return boxes, classes, width, height
 
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, classes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
        return masks, np.asarray(classes, dtype='int32')
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
   ```
  ##   Create Training & Validation Set ##
  ```
  from google.colab import drive
import os
dataset_train = Additive_Manu_Dataset()
dataset_train.load_dataset('', is_train=True)
dataset_train.prepare()
print('Train: %d' % len(dataset_train.image_ids))
 
# test/val set
dataset_val = Additive_Manu_Dataset()
dataset_val.load_dataset('/gdrive/MyDrive/tensorflow1/additive_manufacturing/data/test_labels', is_train=False)
dataset_val.prepare()
print('Test: %d' % len(dataset_val.image_ids))
  ```
 ## Load random samples ##
 ```
 image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
   ```
  ## Train ##
  ```
 
train = ImageDataGenerator(rescale=(1/255))
validation = ImageDataGenerator(rescale=(1/255))

test_generator = ImageDataGenerator(rescale=(1/255))

t = train.flow_from_directory(r'/gdrive/MyDrive/tensorflow1/additive_manufacturing/data/images/',
                              target_size= (200,200),
                              batch_size=3,
                              class_mode="binary")
y = train.flow_from_directory(r'/gdrive/MyDrive/tensorflow1/additive_manufacturing/data/test/',
                              target_size= (200,200),
                              batch_size=3,
                              class_mode="binary")
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation= "relu", input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3), activation= "relu", input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3), activation= "relu", input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(128,(3,3), activation= "relu", input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(512,(3,3), activation= "relu", input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation="relu"),
                                    tf.keras.layers.Dense(1,activation="sigmoid"),
                                    ])
model.compile(loss="binary_crossentropy",
              optimizer = RMSprop(lr=0.001),
              metrics =["accuracy"])

model_fit = model.fit(t,steps_per_epoch = 3, epochs = 10000 ,validation_data= y)

  ```
  ## Test ##
  ```
  img = image.load_img(r'/gdrive/MyDrive/tensorflow1/additive_manufacturing/data/test_labels/13.jpg',target_size= (200,200))
  plt.imshow(img)
  plt.show()
  X = img
X = np.expand_dims(X,axis=0)
images = np.vstack([X])
val = model.predict(images)
if val == 1:
    print("Layer 1")
elif val== 2:
  print("Layer 2")
elif val== 3:
  print("Layer 3")
elif val== 4:
  print("Layer 4")
elif val== 5:
  print("Layer 5")
elif val== 6:
  print("Layer 6")
elif val== 7:
  print("Layer 7")
elif val== 8:
  print("Layer 8")
elif val== 9:
  print("Layer 9")
elif val== 10:
  print("Layer 10")
elif val== 11:
  print("Layer 11")
else:
    print("defect")
                                      
  ```
