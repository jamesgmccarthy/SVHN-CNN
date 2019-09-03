# adapted from https://github.com/pitsios-s/SVHN-Thesis/blob/master/src/multi_digit/svhn.py
import h5py
import os
import numpy as np
import PIL.Image as Image


class ImageProcessor:
    def __init__(self, output_dir, max_labels=5, normalise=True, grey=True):
        self.PIXEL_DEPTH = 255
        self.NUM_LABELS = 11
        self.OUT_HEIGHT = 64
        self.OUT_WIDTH = 64
        self.OUT_CHANNELS = 3
        self.max_labels = max_labels
        self.output_dir = output_dir
        self.file = None
        self.digit_struct_name = None
        self.digit_struct_bbox = None
        self.normalise = normalise
        self.greyscale = grey

    def get_image_name(self, n):
        """Return current image's name
        Args:
            n: The index of image

        Returns:
            The name of the image
        """
        name = ''.join([chr(char)
                        for char in self.file[self.digit_struct_name[n][0]].value])
        return name

    def bounding_box_att_extract(self, att):
        """Returns the specified attribute from the .mat bounding box struct
        Args:
            att: The attribute to extract
        Returns:
            The value of the attribute
        """
        if len(att) > 1:
            attribute = [self.file[att.value[i].item()].value[0][0]
                         for i in range(len(att))]
        else:
            attribute = [att.value[0][0]]
        return attribute

    def get_bounding_box(self, n):
        """Extracts and returns the bounding box info, regarding the n-th image

        Args:
            n: index of the image

        Returns:
            Bounding box information about the n-th image
        """
        bbox = {}
        bb = self.digit_struct_bbox[n].item()
        bbox['label'] = self.bounding_box_att_extract(self.file[bb]['label'])
        bbox['top'] = self.bounding_box_att_extract(self.file[bb]['top'])
        bbox['left'] = self.bounding_box_att_extract(self.file[bb]['left'])
        bbox['height'] = self.bounding_box_att_extract(self.file[bb]['height'])
        bbox['width'] = self.bounding_box_att_extract(self.file[bb]['width'])
        return bbox

    def get_digit_structure(self, n):
        """Creates and returns the whole structure of the image, based on bbox

        Args:
            n: The index of image

        Returns:
            Image structure
        """
        structure = self.get_bounding_box(n)
        structure['name'] = self.get_image_name(n)
        return structure

    def get_all_images_and_digit_structure(self):
        """Loops through images and returns array containing structure of each image
        """
        structures = []
        for i in range(len(self.digit_struct_name)):
            structure = self.get_digit_structure(i)
            if len(structure['label']) <= self.max_labels:
                structures.append(structure)
        return structures

    def save_data(self, original, cropped, labels, name):
        """Saves image data and labels in hdf5 format
        """
        # only create files if not created already
        if not os.path.isfile(name+"_original.h5"):
            original_output_file = h5py.File(os.path.join(
                self.output_dir, name + "_original.h5"), 'w')
            original_output_file.create_dataset(
                name + "_dataset", data=original)
            original_output_file.create_dataset(name+'_labels', data=labels)
        if not os.path.isfile(name+"_cropped.h5"):
            cropped_output_file = h5py.File(os.path.join(
                self.output_dir, name + "_cropped.h5"), 'w')
            cropped_output_file.create_dataset(name + "_dataset", data=cropped)
            cropped_output_file.create_dataset(name+"_labels", data=labels)

    def load_data(self, name, dataset):
        """Loads an hdf5 file that contains the image data and labels

        Args:
            name: the name of the image file

        Returns:
            The data and labels arrays
        """
        input_file = h5py.File(os.path.join(
            self.output_dir, name+"_"+dataset+".h5"), 'r')
        data = input_file[name+'_dataset'][:]
        labels = input_file[name+'_labels'][:]
        return data, labels

    def read_digit_struct(self, data_dir):
        """Reads the digitStruct file and returns the name of 
        the image along with its corresponding bounding box
        """
        struct_file = data_dir + '/digitStruct.mat'
        self.file = h5py.File(struct_file, 'r')
        self.digit_struct_name = self.file['digitStruct']['name']
        self.digit_struct_bbox = self.file['digitStruct']['bbox']
        structs = self.get_all_images_and_digit_structure()
        return structs

    def process_file(self, data_dir):
        """Process all images one by one and return them, together with their labels

        Args:
            data_dir: The directory that contains all the images and the structure file

        Returns:
            The processed images and their labels
        """
        structs = self.read_digit_struct(data_dir)
        data_count = len(structs)

        image_data = np.zeros((data_count, self.OUT_HEIGHT,
                               self.OUT_WIDTH, self.OUT_CHANNELS), dtype=np.float32)
        cropped_data = np.zeros((data_count, self.OUT_HEIGHT,
                                 self.OUT_WIDTH, self.OUT_CHANNELS), dtype=np.float32)
        labels = np.zeros((data_count, self.max_labels,
                           self.NUM_LABELS), dtype=np.int32)

        for i in range(data_count):
            label = structs[i]['label']
            file_name = os.path.join(data_dir, structs[i]['name'])
            top = structs[i]['top']
            left = structs[i]['left']
            height = structs[i]['height']
            width = structs[i]['width']

            labels[i] = self.create_label_array(label)
            image_data[i], cropped_data[i] = self.create_image_array(
                file_name, top, left, height, width)
        return image_data, cropped_data, labels

    def create_label_array(self, labels):
        """Creates the label array
        """
        num_digits = len(labels)
        labels_array = np.ones([self.max_labels], dtype=np.int32)*10
        one_hot_labels = np.zeros(
            (self.max_labels, self.NUM_LABELS), dtype=np.int32)

        for n in range(num_digits):
            if labels[n] == 10:
                labels[n] = 0
            labels_array[n] = labels[n]

        for n in range(len(labels_array)):
            one_hot_labels[n] = self.one_hot_encode(labels_array[n])

        return one_hot_labels

    def one_hot_encode(self, number):
        one_hot = np.zeros(shape=self.NUM_LABELS, dtype=np.int32)
        one_hot[number] = 1

        return one_hot

    def create_image_array(self, file_name, top, left, height, width):
        # Load image
        image = Image.open(file_name)

        # Find the top corner of bounding box
        image_top = np.amin(top)

        # Find left corner of bounding box
        image_left = np.amin(left)
        image_height = np.amax(top) + height[np.argmax(top)] - image_top
        image_width = np.amax(left) + width[np.argmax(left)] - image_left

        # Find smallest possible bounding box + expand by 30%
        # following advice by Ian Goodfellow et al.
        bbox_left = np.floor(image_left - 0.1 * image_width)
        bbox_top = np.floor(image_top - 0.1 * image_height)
        bbox_right = np.amin(
            [np.ceil(bbox_left + 1.3 * image_width), image.size[0]])
        bbox_bottom = np.amin([
            np.ceil(image_top + 1.3 * image_height), image.size[1]])

        # Create cropped image
        cropped_image = image.crop((bbox_left, bbox_top, bbox_right, bbox_bottom)).resize(
            [self.OUT_HEIGHT, self.OUT_WIDTH])

        # Resize original
        image = image.resize([self.OUT_HEIGHT, self.OUT_WIDTH])

        # Convert original image and cropped image to np array
        image_array = np.array(image)
        cropped_array = np.array(cropped_image)

        return image_array, cropped_array


def main():
    image_processor = ImageProcessor(
        "./Data/processed", max_labels=5, normalise=True, grey=False)

    # Training set
    if not os.path.isfile("./Data/processed/train_cropped.h5"):
        print("Making directory: ./Data/processed")
        if not os.path.isdir("./Data/processed"):
            os.mkdir("./Data/processed")
        print("Making directory ./Data/Images/train")
        if not os.path.isdir("./Data/Images"):
            os.mkdir("./Data/Images")
        if not os.path.isdir("./Data/Images/train"):
            os.mkdir('./Data/Images/train')
        train_original, train_cropped, train_labels = image_processor.process_file(
            './Data/train')
        image_processor.save_data(
            train_original, train_cropped, train_labels, 'train')

    # Test set
    if not os.path.isfile("./Data/processed/test_cropped.h5"):
        if not os.path.isdir('./Data/Images/test'):
            print("Making directory ./Data/Images/test")
            os.mkdir('./Data/Images/test')
        test_original, test_cropped, test_labels = image_processor.process_file(
            './Data/test')
        image_processor.save_data(
            test_original, test_cropped, test_labels, 'test')


if __name__ == "__main__":
    main()
