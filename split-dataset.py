from sklearn.model_selection import train_test_split
import shutil
import os
from PIL import Image

source_dataset_path = "datasets/kagglecatsanddogs/PetImages"
target_path = "datasets/catsanddogs"

test_ratio = 0.2
val_ratio = 0.2


#Image Folder
# It assumes that the format of label is as follows:
# cats/photo1.png
# cats/photo2.png
# dogs/photo1.png
# dogs/photo2.png
annotation_type = "ImageFolder"
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

if not os.path.exists(source_dataset_path):
    raise Exception("Your path {} does not exists.. Check it again.".format(source_dataset_path))

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def is_valid_image(filename):
    if is_image_file(filename):
        try:
            img = Image.open(filename)
            img.verify()
            return True
        except (IOError, SyntaxError):
            print ("Bad file: ", filename)
            return False
    else:
        return False
    

def find_class(directory):
    classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def make_dataset(directory, class_to_idx, extensions=None):
    images = []
    targets = []
    directory = os.path.expanduser(directory)

    #Check if extension is valid
    # if extensions is not None:
    #     def is_valid_file(x):
    #         return(has_file_allowed_extension(x,extensions))

    for label in class_to_idx.keys():
        full_dir = os.path.join(directory, label)
        if not os.path.isdir(full_dir):
            continue

        for root, _, files in os.walk(full_dir):
            for fname in sorted(files):
                full_path = os.path.join(root,fname)
                if not is_valid_image(full_path): #TODO: Make it more general file
                    continue
                images.append(full_path)
                targets.append(class_to_idx[label])

    return images, targets
    
def split_dataset(dataset_path):
    print("Splitting dataset into train,val,test split...")
    labels, class_to_idx = find_class(dataset_path)
    images, targets = make_dataset(dataset_path, class_to_idx, IMG_EXTENSIONS)

    raw_X_train, X_test, raw_Y_train, Y_test = train_test_split(images, targets, test_size=test_ratio, random_state=24, stratify=targets)
    X_train, X_val, Y_train, Y_val = train_test_split(raw_X_train,raw_Y_train, test_size=val_ratio, random_state=24, stratify=raw_Y_train)

    return labels, {'train':(X_train, Y_train),'val':(X_val, Y_val), 'test':(X_test, Y_test)}

def create_dir_if_not_exists(target_dir):
    if not os.path.isdir(target_dir):
        print ("Creating {}".format(target_dir))
        os.mkdir(target_dir)

# Move dataset
def move_dataset(split_set, labels, target_path):
    create_dir_if_not_exists(target_path)

    for setname, dataset in split_set.items():
        print("Building {}...".format(setname))
        current_set = os.path.join(target_path, setname)

        create_dir_if_not_exists(current_set)

        [create_dir_if_not_exists(os.path.join(current_set,label)) for label in labels]

        for fname, idx in zip(*dataset):
            if os.path.isfile(fname):
                target_full_path = os.path.join(current_set,labels[idx],os.path.basename(fname))
                shutil.copy(fname,target_full_path)
        
        print("Successfully built {}...".format(setname))


labels, split_set = split_dataset(source_dataset_path)
move_dataset(split_set,labels,target_path)




