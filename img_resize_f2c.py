# http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287

from argparse import ArgumentParser
import os
from scipy import misc
import numpy as np

from PIL import Image
from tqdm import tqdm

import pickle

val_names_file = 'val.txt'
val_labels_file = 'ILSVRC2015_clsloc_validation_ground_truth.txt'
map_file = 'map_clsloc.txt'


alg_dict = {
    'lanczos': Image.Resampling.LANCZOS,
    'nearest': Image.Resampling.NEAREST,
    'bilinear': Image.Resampling.BILINEAR,
    'bicubic': Image.Resampling.BICUBIC,
    'hamming': Image.Resampling.HAMMING,
    'box': Image.Resampling.BOX
}

# Return dictionary where key is validation image name and value is class label
# ILSVRC2012_val_00000001: 490
# ILSVRC2012_val_00000002: 361
# ILSVRC2012_val_00000003: 171
# ...
def get_val_ground_dict():
    # Table would be better? but keep dict
    d_labels = {}
    i = 1
    with open(val_labels_file) as f:
        for line in f:
            tok = line.split()
            d_labels[i] = int(tok[0])
            i += 1

    d = {}
    with open(val_names_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = d_labels[int(tok[1])]
    return d


# Get list of folders with order as in map_file
# Useful when we want to have the same splits (taking every n-th class)
def get_ordered_folders():
    folders = []

    with open(map_file) as f:
        for line in f:
            tok = line.split()
            folders.append(tok[0])
    return folders


# Returns dictionary where key is folder name and value is label num as int
# n02119789: 1
# n02100735: 2
# n02110185: 3
# ...
def get_label_dict():
    d = {}
    with open(map_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = int(tok[1])
    return d


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images")
    parser.add_argument('-o', '--out_dir', help="Output directory for pickle files")
    parser.add_argument('-alg', '--resize_algo', help="Which resizing algorithm to use", default="bicubic")
    parser.add_argument('-s', '--size', help="Resizing to what size")
    args = parser.parse_args()

    return args.in_dir, args.out_dir, args.resize_algo, args.size

def make_dataset(in_dir, max_images_count=float('inf')):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.tif', '.TIF', '.tiff', '.TIFF',
    ]

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
        
    images = []
    assert os.path.isdir(in_dir), '%s is not a valid directory' % in_dir

    for root, _, fnames in sorted(os.walk(in_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    return images[:min(max_images_count, len(images))]

def str2alg(str):
    str = str.lower()
    return alg_dict.get(str, None)

# Strong assumption about in_dir and out_dir (They must contain proper data)
def process_folder(in_dir, out_dir, alg: str, img_size: int, max_images_count=float('inf')):
    # label_dict = get_label_dict()
    # folders = get_ordered_folders()

    # Here subsample folders (If desired) [1*]
    # folders = folders[0::10]
    # folders = folders[900:902]
    
    img_size = int(img_size)
    alg_val = str2alg(alg)
    
    if alg_val is None:
        print("Sorry but this algorithm (%s) is not available, use help for more info." % alg)
        return
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print("Processing folder %s" % in_dir)
    # data_list_train = []
    # labels_list_train = []
    # images_A, images_B = [], []
    
    images_paths = make_dataset(in_dir, max_images_count)
    
    # for image_name in os.listdir(in_dir): # OLD
    # for image_name in images_paths:
    for image_name in tqdm(images_paths, desc="Processing images"):
        # label = label_dict[folder]
        # print("Processing images from folder %s as label %d" % (folder, label))
        # Get images from this folder
        
        # for image_name in os.listdir(os.path.join(in_dir, folder)):
        try:
            # OLD:
            # img = misc.imread(os.path.join(in_dir, image_name)) 
            # r = img[:, :, 0].flatten()
            # g = img[:, :, 1].flatten()
            # b = img[:, :, 2].flatten()
            
            AB = Image.open(image_name).convert('RGB')
            # split AB image into A and B
            w, h = AB.size
            # w2 = int(w / 2)
            w2 = w
            
            A = AB.crop((0, 0, w2, h))
            
            A = A.resize((img_size, img_size), alg_val)
                                    
            new_AB = Image.new(mode='RGB', size=(img_size, img_size), color=(255, 255, 255))
            new_AB.paste(A, (0, 0))
            
            filename = os.path.splitext(image_name)[0].split('/')[-1]
            new_AB.save(os.path.join(out_dir, filename + '.png'))
            
        except:
            print('Cant process image %s' % image_name)
            with open("log_img2np.txt", "a") as f:
                f.write("Couldn't read: %s \n" % os.path.join(in_dir, image_name))
            # continue
            exit()

if __name__ == '__main__':
    in_dir, out_dir, alg, img_size = parse_arguments()

    print("Starting the program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir, alg=alg, img_size=img_size)
    print("Finished.")