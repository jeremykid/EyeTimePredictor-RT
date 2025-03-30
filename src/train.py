import torch.nn as nn
import torch

# Training loop
def patch_tst_train_regression_one_epoch(model, dataloader, optimizer, lossfun = nn.BCEWithLogitsLoss(), device = torch.device('cpu')):
    model.train()
    total_loss = 0.0
    for xlsx_name, X_batch, Y_batch in dataloader:
        outputs = model(past_values=X_batch.to(device))
        outputs = outputs['regression_outputs']
        loss = lossfun(outputs, Y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validate loop
def patch_tst_validation_regression_one_epoch(model, dataloader, lossfun = nn.BCEWithLogitsLoss(), device = torch.device('cpu')):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xlsx_name, X_batch, Y_batch in dataloader:
            outputs = model(past_values=X_batch.to(device))
            outputs = outputs['regression_outputs']
            loss = lossfun(outputs, Y_batch.to(device))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Training loop
def patch_tst_train_segmentation_one_epoch(model, dataloader, optimizer, lossfun = nn.BCEWithLogitsLoss(), device = torch.device('cpu')):
    model.train()
    total_loss = 0.0
    for xlsx_name, X_batch, Y_batch in dataloader:
        outputs = model(past_values=X_batch.to(device))
        outputs = outputs['prediction_logits']
        loss = lossfun(outputs, Y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validate loop
def patch_tst_validation_segmentation_one_epoch(model, dataloader, lossfun = nn.BCEWithLogitsLoss(), device = torch.device('cpu')):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xlsx_name, X_batch, Y_batch in dataloader:
            outputs = model(past_values=X_batch.to(device))
            outputs = outputs['prediction_logits']
            loss = lossfun(outputs, Y_batch.to(device))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss