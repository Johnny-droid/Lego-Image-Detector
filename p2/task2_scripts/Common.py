import os
import cv2
import numpy as np
import ast
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import display, Image

np.random.seed(42)

def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

def loadImage(num, imgLoadSizeRatio = 1, dataDir = 'data/imgs', standardSize = -1):
    img = cv2.imread(os.path.join(dataDir, f'{num}'))
    if standardSize > 0:
        img = cv2.resize(img, (standardSize, standardSize))
    elif imgLoadSizeRatio != 1:
        img = cv2.resize(img, (0, 0), fx = imgLoadSizeRatio, fy = imgLoadSizeRatio)
    return img

def render(image):
    if image.dtype == np.float64:
        image = cv2.convertScaleAbs(image)
    if len(image.shape) == 3 and image.shape[2] == 3: # BGR or RGB
        if np.array_equal(image[:, :, 0], image[:, :, 2]):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_bytes = cv2.imencode('.png', image)[1].tobytes()
    display(Image(data=img_bytes))



def getActualPieceCount(imgID, df_lego_sets = None):
    piece_count = df_lego_sets.loc[df_lego_sets['id'] == imgID, 'lego_count'].values[0]
    return piece_count 

def getBoundingBoxes(name, df_lego_sets = None):
    bb_list_str = df_lego_sets.loc[df_lego_sets['id'] == name, 'bb_list'].values[0]
    bb_list = ast.literal_eval(bb_list_str)
    return bb_list

def makeGuess(image_id, num_guess):
    piece_count = getActualPieceCount(image_id)
    num_legos_error = abs(num_guess - piece_count)
    
    if(num_legos_error > 0):
        print(f"Error in Lego Count - Guessed: {num_guess} | Actual: {piece_count} legos")
    else :
        print(f"Perfect ({num_guess}) Guess!")
        
    return piece_count, num_legos_error

def getAllImages(datasetSize = 200, DRIVE_ROOT_DIR = "", limit=True):
    imgs = []

    df_lego_sets = pd.read_csv(DRIVE_ROOT_DIR + "data/values.csv")
    img_names = get_file_names(DRIVE_ROOT_DIR + 'data/imgs')

    for name in img_names:
        img = loadImage(name, 1, DRIVE_ROOT_DIR + 'data/imgs')
        count = getActualPieceCount(name[:-4], df_lego_sets)
        bbs = getBoundingBoxes(name[:-4], df_lego_sets)
        imgs.append((img, count, bbs))

        if (count < 2):
            continue
        # apply multiple transformations to the image to upsample the dataset
        # Rotation
        rotated_image_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        imgs.append((rotated_image_90, count))
        
        rotated_image_180 = cv2.rotate(img, cv2.ROTATE_180)
        imgs.append((rotated_image_180, count))
        
        rotated_image_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgs.append((rotated_image_270, count))
        
        # Flipping
        flipped_image = cv2.flip(img, flipCode=1)
        imgs.append((flipped_image, count))
        
        flipped_image = cv2.flip(img, flipCode=0)
        imgs.append((flipped_image, count))
        
        # Rotation + Flipping
        rotated_flipped_image = cv2.flip(rotated_image_90, flipCode=1)
        imgs.append((rotated_flipped_image, count))

    # From the images that have just 1 lego, remove half of them
    ones_imgs = [img for img in imgs if img[1] == 1]
    np.random.shuffle(ones_imgs)
    ones_imgs = ones_imgs[:int(len(ones_imgs)/2)]
    imgs = [img for img in imgs if img[1] != 1]
    imgs.extend(ones_imgs)

    # Display the balance of the dataset
    print("Proportion of images for whole dataset")
    lego_counts = [img[1] for img in imgs]
    lego_counts = Counter(lego_counts)
    print(lego_counts)
    print()

    # suffle the images
    np.random.shuffle(imgs)

    if limit:
        imgs = imgs[:datasetSize]

        
    # pick the first 70% of the images for training
    train_imgs = imgs[:int(len(imgs)*0.7)]

    # pick the remaining 15% of the images for validation and the last 15% for testing
    val_imgs = imgs[int(len(imgs)*0.7):int(len(imgs)*0.85)]
    test_imgs = imgs[int(len(imgs)*0.85):]

    print("\nSizes of datasets:\nTrain: ", len(train_imgs), "\nValidation: ", len(val_imgs), "\nTest: ", len(test_imgs))

    # Calculate the distribution of the number of legos in each set
    train_lego_counts = [img[1] for img in train_imgs]
    val_lego_counts = [img[1] for img in val_imgs]
    test_lego_counts = [img[1] for img in test_imgs]

    train_lego_counts = Counter(train_lego_counts)
    val_lego_counts = Counter(val_lego_counts)
    test_lego_counts = Counter(test_lego_counts)

    print()
    print("Train: ", train_lego_counts)
    print("Validation: ", val_lego_counts)
    print("Test: ", test_lego_counts)

    return train_imgs, val_imgs, test_imgs

def plotTrainingHistory(train_history, val_history, stat='accuracy'):
    plt.subplot(2, 1, 1)
    plt.title('Loss')
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title(stat)
    plt.plot(train_history[stat], label='train')
    plt.plot(val_history[stat], label='val')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


# Display the confusion matrix exactly as received from the sklearn library
def display_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # We will have at most 33 classes
    plt.xticks(range(0, 32))
    plt.yticks(range(0, 32))
    # The padding should be set to 0
    plt.gca().set_xticklabels([str(x) for x in range(1, 33)])
    plt.gca().set_yticklabels([str(x) for x in range(1, 33)])
    plt.show()

def buildConfusionMatrix(preds, labels):
    # We will build our custom confusion matrix with 32 classes
    confusion_matrix = np.zeros((32, 32))
    for i in range(len(preds)):
        confusion_matrix[labels[i]-1, preds[i]-1] += 1
    return confusion_matrix


def showErrors(model, dataloader, num_examples=20, device="cpu"):    
    plt.figure(figsize=(15, 15))

    for ind, (X, y) in enumerate(dataloader):
      if ind >= num_examples: break
      X, y = X.to(device), y.to(device)    
      pred = model(X)
      probs = F.softmax(pred, dim=1)
      final_pred = torch.argmax(probs, dim=1)

      plt.subplot(10, 10, ind + 1)
      plt.axis("off")
      plt.text(0, -1, y[0].item(), fontsize=14, color='green') # correct
      plt.text(8, -1, final_pred[0].item(), fontsize=14, color='red')  # predicted
      plt.imshow(X[0][0,:,:].cpu(), cmap='gray')
      
    plt.show()