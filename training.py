import argparse
import os.path
from ProtSeqGen_code.Model.datasets import *

def main(args):
    """Main training routine for ProtseqGen model on CATH dataset.

    This function handles:
        - Dataset loading and preprocessing
        - Model initialization and checkpoint loading
        - Training loop with optional mixed-precision
        - Validation and test evaluation
        - Periodic checkpoint saving
    """
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from ProtseqGen_code.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProtseqGen, worker_init_fn, get_pdbs
    from ProtseqGen_code.datasets import cath_dataset

    # Setup scaler and device
    scaler = torch.amp.GradScaler('cuda')
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Prepare output directories
    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    PATH = args.previous_checkpoint

    # Initialize log file
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, 
        "HOMO"    : 0.70 
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    # Load datasets and create DataLoaders
    train_set, valid_set, test_set = cath_dataset(1800, jsonl_file='../data/chain_set.jsonl')
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    # Initialize model and move to device
    model = ProtseqGen(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)

    # Load checkpoint if exists
    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] 
        epoch = checkpoint['epoch'] 
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    # Initialize optimizer
    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Preload PDBs with multiprocessing
    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        z = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            z.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
        pdb_dict_test = z.get().result()
        
        loader_train = pdb_dict_train
        loader_valid = pdb_dict_valid
        loader_test = pdb_dict_test

        # Training loop
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            # Reload data periodically
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    loader_train = pdb_dict_train
                    pdb_dict_valid = p.get().result()
                    loader_valid = pdb_dict_valid
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1

            # Iterate through training batches    
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask = batch
                # squeeze batch dimension
                X = X.squeeze(0)
                S = S.squeeze(0)
                mask = mask.squeeze(0)
                chain_M = torch.ones_like(S)
                X = X.to(device)
                S = S.to(device).long()
                mask = mask.to(device)
                chain_M = chain_M.to(device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                # Forward and backward pass
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    scaler.scale(loss_av_smoothed).backward()
    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
    
                    optimizer.step()

                # Compute loss and accuracy
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            #Perform validation, checkpoint saving, and test evaluation for one epoch
            model.eval()
            with torch.no_grad():
                # Initialize validation accumulators
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                # Loop over validation batches
                for _, batch in enumerate(loader_valid):
                    X, S, mask = batch
                    # Remove batch dimension
                    X = X.squeeze(0)
                    S = S.squeeze(0)
                    mask = mask.squeeze(0)
                    chain_M = torch.ones_like(S)
                    X = X.to(device)
                    S = S.to(device).long()
                    mask = mask.to(device)
                    chain_M = chain_M.to(device)
                    # Forward pass
                    log_probs = model(X, S, mask, chain_M)
                    mask_for_loss = mask*chain_M

                    # Compute loss and accuracy
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                    # Accumulate metrics
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            # Compute training and validation metrics           
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            # Format metrics for logging
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=4)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=4)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=4)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=4)

            # Log epoch time and metrics
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            
            # Define checkpoint path
            best_validation_loss = float("inf")  
            best_model_path = base_folder + 'model_weights/best_model.pt'
            checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'

            # Save last epoch checkpoint
            torch.save({
                'epoch': e+1,
                'step': total_step,
                'num_edges' : args.num_neighbors,
                'noise_level': args.backbone_noise,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
            }, checkpoint_filename_last)

            # Save best model if validation improved
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss  
                torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, best_model_path)
                print(f'Best model updated at epoch {e+1}, step {total_step} with validation loss {validation_loss:.4f}')

            # Periodically save checkpoints
            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)
                
                
            #Perform test evaluation for the epoch              
            model.eval()
            with torch.no_grad():
                # Initialize test accumulators
                test_sum, test_weights = 0., 0.
                test_acc = 0.
                # Loop over test batches
                for _, batch in enumerate(loader_test):
                    X, S, mask = batch
                    X = X.squeeze(0)
                    S = S.squeeze(0)
                    mask = mask.squeeze(0)
                    chain_M = torch.ones_like(S)
                    X = X.to(device)
                    S = S.to(device).long()
                    mask = mask.to(device)
                    chain_M = chain_M.to(device)
                    log_probs = model(X, S, mask, chain_M)
                    mask_for_loss = mask * chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                    test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    test_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                # Compute test metrics
                test_loss = test_sum / test_weights
                test_accuracy = test_acc / test_weights
                test_perplexity = np.exp(test_loss)

                # Format metrics
                test_perplexity_ = np.format_float_positional(np.float32(test_perplexity), unique=False, precision=4)
                test_accuracy_ = np.format_float_positional(np.float32(test_accuracy), unique=False, precision=4)
                
                # Log test results
                with open(logfile, 'a') as f:
                    f.write(f'epoch: {e+1}, step: {total_step}, test_perplexity: {test_perplexity_}, test_acc: {test_accuracy_}\n')
                print(f'epoch: {e+1}, step: {total_step}, test_perplexity: {test_perplexity_}, test_acc: {test_accuracy_}')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=800, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.32, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
 
    args = argparser.parse_args()    
    main(args)   
