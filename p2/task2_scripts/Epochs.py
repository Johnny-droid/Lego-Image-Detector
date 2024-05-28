import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

np.random.seed(42)

############################################ Classification ############################################

def epoch_iter_classification(dataloader, model, loss_fn, optimizer=None, is_train=True, preds=[], labels=[], device="cpu"):
    if is_train:
      assert optimizer is not None, "When training, please provide an optimizer."

    num_batches = len(dataloader)
    
    if num_batches == 0:
      print("No data in the dataloader")
      return 0.0, 0.0

    if is_train:
      model.train()
    else:
      model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
      for batch, (X, y) in enumerate(tqdm(dataloader)):
          X, y = X.to(device), y.to(device)

          # Obtain prediction
          pred = model(X)
          
          # Obtain loss value
          loss = loss_fn(pred, y)

          if is_train:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          # Save training metrics
          total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached

          # Calculate final prediction
          probs = F.softmax(pred, dim=1)
          final_pred = torch.argmax(probs, dim=1)
          preds.extend(final_pred.cpu().numpy())
          labels.extend(y.cpu().numpy())

    return total_loss / num_batches, accuracy_score(labels, preds)


def train_classification(model, model_name, num_epochs, train_dataloader, validation_dataloader, loss_fn, optimizer, directory="", device="cpu"):
  
  train_history = {'loss': [], 'accuracy': []}
  val_history = {'loss': [], 'accuracy': []}
  
  best_val_loss = np.inf
  
  print("Start training...")
  
  for t in range(num_epochs):
    
      print(f"\nEpoch {t+1}")
      
      train_loss, train_acc = epoch_iter_classification(train_dataloader, model, loss_fn, optimizer, device=device)
      print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")
      
      if(train_acc > 0):
        val_loss, val_acc = epoch_iter_classification(validation_dataloader, model, loss_fn, is_train=False, device=device)
      else :
        val_loss = 0
        val_acc = 0
        
      print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

      # save model when val loss improves
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, directory + "models/" + model_name + '_best_model.pth')

      # save latest model
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, directory + "models/" + model_name + '_latest_model.pth')

      # save training history for plotting purposes
      train_history["loss"].append(train_loss)
      train_history["accuracy"].append(train_acc)

      val_history["loss"].append(val_loss)
      val_history["accuracy"].append(val_acc)
      
  print("Finished")
  return train_history, val_history



############################################ Regression ############################################

# We need to change the epoch_iter to adapt to the regression problem instead of classification

def epoch_iter_regression(dataloader, model, loss_fn, optimizer=None, is_train=True, preds=[], labels=[], device="cpu"):
    if is_train:
      assert optimizer is not None, "When training, please provide an optimizer."

    num_batches = len(dataloader)
    
    if num_batches == 0:
      print("No data in the dataloader")
      return 0.0, 0.0

    if is_train:
      model.train()
    else:
      model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
      for batch, (X, y) in enumerate(tqdm(dataloader)):
          X, y = X.to(device), y.to(device)

          y = y.float().unsqueeze(1)

          # Obtain prediction
          pred = model(X)
          
          # Obtain loss value
          loss = loss_fn(pred, y)

          if is_train:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          # Save training metrics
          total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached

          # Calculate final prediction
          preds.extend(pred.cpu().detach().numpy())
          labels.extend(y.cpu().detach().numpy())

    return total_loss / num_batches


def train_regression(model, model_name, num_epochs, train_dataloader, validation_dataloader, loss_fn, optimizer, directory="", device="cpu"):
    
    train_history = {'loss': [], 'mse': []}
    val_history = {'loss': [], 'mse': []}
    
    best_val_loss = np.inf
    
    print("Start training...")
    
    for t in range(num_epochs):
        
        print(f"\nEpoch {t+1}")
        
        train_loss = epoch_iter_regression(train_dataloader, model, loss_fn, optimizer, device=device)
        print(f"Train loss: {train_loss:.3f}\t")
        
        if(train_loss > 0):
            val_loss = epoch_iter_regression(validation_dataloader, model, loss_fn, is_train=False, device=device)
        else :
            val_loss = 0
            
        print(f"Val loss: {val_loss:.3f} \t")
    
        # save model when val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, directory + "models/" + model_name + '_best_model.pth')
    
        # save latest model
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, directory + "models/" + model_name + '_latest_model.pth')
    
        # save training history for plotting purposes
        train_history["loss"].append(train_loss)
    
        val_history["loss"].append(val_loss)
        
    print("Finished")
    return train_history, val_history