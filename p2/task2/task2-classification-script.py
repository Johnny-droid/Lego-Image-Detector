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
    DRIVE_ROOT_DIR = "../"
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    NUM_EPOCHS = 50

    MODEL_NAME =  "vgg16-classifier" # "cnn-classifier"# "resnet18-classifier"
    MODEL = VGG16Classifier()  #ConvolutionalNeuralNetworkClassifier()  # ResNet18Classifier()
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0001)

    try:
        folder_name = "models"
        os.mkdir(folder_name)
    except FileExistsError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the data
    train_imgs, val_imgs, test_imgs = getAllImages(DATASET_SIZE, DRIVE_ROOT_DIR, DATASET_SET_LIMIT)
    train_dataloader, validation_dataloader, test_dataloader = getAllDataloadersClassification(train_imgs, val_imgs, test_imgs, BATCH_SIZE, NUM_WORKERS)

    # Model
    model = MODEL.to(device)
    loss_fn = LOSS_FN
    optimizer = OPTIMIZER

    # Training
    train_history, val_history = train_classification(model, MODEL_NAME, NUM_EPOCHS, train_dataloader, validation_dataloader, loss_fn, optimizer, "", device)
    plotTrainingHistory(train_history, val_history, stat='accuracy')

    # Test model
    best_model = MODEL.to(device)
    checkpoint = torch.load("models/" + MODEL_NAME + "_best_model.pth")
    best_model.load_state_dict(checkpoint['model'])

    preds, labels = [], []

    print("\n\nTesting the model...")
    test_loss, test_acc = epoch_iter_classification(test_dataloader, best_model, loss_fn, is_train=False, device=device, preds=preds, labels=labels)
    print(f"Test loss: {test_loss:.3f} \t Test acc: {test_acc:.3f}")

    preds = [x+1 for x in preds]
    labels = [x+1 for x in labels]

    cm = buildConfusionMatrix(preds, labels)
    display_confusion_matrix(cm)