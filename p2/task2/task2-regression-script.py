import os
import torch
from task2_scripts import Common, Models, Dataset, Epochs
from task2_scripts.Common import *
from task2_scripts.Models import *
from task2_scripts.Dataset import *
from task2_scripts.Epochs import *

if __name__ == "__main__":
    DATASET_SET_LIMIT = False # Set to False to use the full dataset
    DATASET_SIZE = 50
    DRIVE_ROOT_DIR = ""
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    NUM_EPOCHS = 3

    MODEL_NAME = "cnn-regression" # "vgg16-regression" # "resnet18-regression"
    MODEL = ConvolutionalNeuralNetworkRegression() # VGG16Regression() # ResNet18Regression()
    LOSS_FN = nn.MSELoss()
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.0001)

    try:
        folder_name = DRIVE_ROOT_DIR + "models"
        os.mkdir(folder_name)
    except FileExistsError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the data
    train_imgs, val_imgs, test_imgs = getAllImages(DATASET_SIZE, DRIVE_ROOT_DIR, DATASET_SET_LIMIT)
    train_dataloader, validation_dataloader, test_dataloader = getAllDataLoadersRegression(train_imgs, val_imgs, test_imgs, BATCH_SIZE, NUM_WORKERS)

    # Model
    model = MODEL.to(device)
    loss_fn = LOSS_FN
    optimizer = OPTIMIZER

    # Training
    train_history, val_history = train_regression(model, MODEL_NAME, NUM_EPOCHS, train_dataloader, validation_dataloader, loss_fn, optimizer, DRIVE_ROOT_DIR, device)
    plotTrainingHistory(train_history, val_history, stat='mse')

    # Test model
    best_model = MODEL.to(device)
    checkpoint = torch.load(DRIVE_ROOT_DIR + "models/" + MODEL_NAME + "_best_model.pth")
    best_model.load_state_dict(checkpoint['model'])

    preds, labels = [], []

    print("\n\nTesting the model...")
    test_loss = epoch_iter_regression(test_dataloader, best_model, loss_fn, is_train=False, device=device, preds=preds, labels=labels)
    print(f"Test loss: {test_loss:.3f}")

    print(preds[0])
    print(preds[0][0])
    preds = [round(float(x.squeeze(0))) for x in preds]
    labels = [round(float(x.squeeze(0))) for x in labels]

    # Print How many errors we have
    errors = 0
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            errors += 1
    print(f"Number of errors: {errors}")
    print(f"Total number of images: {len(preds)}")
    print(f"Accuracy: {1 - (errors/len(preds))}")

    cm = buildConfusionMatrix(preds, labels)
    display_confusion_matrix(cm)

