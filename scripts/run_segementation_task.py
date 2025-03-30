import argparse
import sys, os
import glob
from transformers import PatchTSTConfig, PatchTSTForClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
sys.path.append(os.path.abspath('../src/'))
from model import LSTMSequenceLabeler
# from train import train_one_epoch, validate_one_epoch
from train import patch_tst_train_segmentation_one_epoch, patch_tst_validation_segmentation_one_epoch
from data_loader import getDataSet

def main(args):
    device = torch.device('cuda:'+args.GPU)
    xlsx_name_list = glob.glob("../../data_new/*/*.xlsx")
    print (len(xlsx_name_list))
    if args.model == 'PatchTST':
        config = PatchTSTConfig(
            num_input_channels=6,
            num_targets=1500,
            context_length=1500,
            patch_length=50,
            stride=25,
            use_cls_token=True,
            loss = "mse"
        )
        model = PatchTSTForClassification(config=config).to(device)
    elif args.model == 'LSTM': 
        model = LSTMSequenceLabeler(input_dim=6, hidden_dim=128, num_layers=2, dropout_rate=0.3).to(device)
    
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True) # TODO hyperparameter

    train_list, test_list = train_test_split(xlsx_name_list, test_size=0.33, random_state=42)    
    PeriodDataSet = getDataSet(task = 'Segmentation')
    train_dataloader = DataLoader(PeriodDataSet(xlsx_name_list = train_list), batch_size=batch_size, shuffle=True)   
    validation_dataloader = DataLoader(PeriodDataSet(xlsx_name_list = test_list), batch_size=batch_size, shuffle=True) 

    early_stopping_patience = 10
    best_val_loss = float('inf')
    early_stopping_counter = 0

    epochs = 100 # TODO hyperparameter
    log_dict = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }
    lossfun = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss = patch_tst_train_segmentation_one_epoch(model, train_dataloader, optimizer, lossfun = lossfun, device = device)
        val_loss = patch_tst_validation_segmentation_one_epoch(model, validation_dataloader, lossfun = lossfun, device = device)

        # Step the scheduler with validation loss
        scheduler.step(val_loss)
        log_dict['epoch'].append(epoch)
        log_dict['train_loss'].append(train_loss)
        log_dict['val_loss'].append(val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.to('cpu').state_dict(), f'{args.model}_segmentation_model.pth')
            model = model.to(device)
            print('Best model saved with validation loss:', best_val_loss)        
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print('Early stopping triggered')
                break

    log_df = pd.DataFrame.from_dict(log_dict)
    log_df.to_csv(f'{args.model}_segmentation_log.csv')      
    
if __name__ == '__main__':
    # args = generate_parser()
    parser = argparse.ArgumentParser(description="Argument parser for regression experiments.")
    parser.add_argument('--model', type=str, default="PatchTST",
                        choices=["PatchTST", "LSTM"],
                        help="Model name.")
    parser.add_argument('--GPU', type=str, default="7",
                        help="GPU device number.")    
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch Size device number.")        
    args = parser.parse_args()
    main(args)