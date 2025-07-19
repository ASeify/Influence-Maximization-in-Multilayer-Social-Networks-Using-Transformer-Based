# Version 0.0.1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics.regression import MeanSquaredError, R2Score, MeanAbsoluteError

import sys
import os
from tqdm import tqdm
from datetime import datetime
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors as bcolors
from Average_Meter_Class import AverageMeter
from Files_Handler_Class import Files_Handler

file_handler_obj = Files_Handler()

class Neural_Network_Functions:

    @staticmethod
    def get_device():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.empty_cache()
        return device

    @staticmethod
    def get_load_model_and_optimazer(loaded_model:str, optimizer_name:str='Adam',
                                     manual_select_model:bool=False, task_type:str='regression'):
        loaded_model_params = {}
        load_model_status = False
        loaded_model_info = None
        if manual_select_model:
            loaded_model = file_handler_obj.select_files("Model file", ".pt", False)
        
        if loaded_model != '':
            try:
                device = Neural_Network_Functions.get_device()
                try:
                    model = torch.load(loaded_model, map_location=torch.device(device), weights_only=False)
                    load_model_status = True
                    loaded_model_info = file_handler_obj.get_file_path_info(loaded_model)
                    try:
                        optimizer_file_path = loaded_model_info['path']
                        optimizer_file = ''
                        mode_name_items = loaded_model_info['name'].strip().split(' ')
                        for i, item in enumerate(mode_name_items):
                            if item != 'model':
                                optimizer_file += item.strip()
                            else:
                                optimizer_file += optimizer_name.strip()
                            if i != (len(mode_name_items)-1):
                                optimizer_file += ' '
                        optimizer_file += '.optim'
                        optimizer_file_path += optimizer_file
                        optimizer = torch.load(optimizer_file_path, map_location=torch.device(device), weights_only=False)
                    except Exception as e:
                        print(bcolors.FAIL + f"Load optimizer fail. {e}" + bcolors.ENDC)
                        return e
                except Exception as e:
                    print(bcolors.FAIL + f"Load model fail. {e}" + bcolors.ENDC)
                    return e
                
                model_name = loaded_model_info['name'].strip()
                model_name_items = model_name.split(' ')
                if task_type == 'classification':
                    model_infos = ['lr', 'wd', 'epochs', 'loss_valid', 'loss_train', 'acc_valid', 'acc_train']
                    for item in model_name_items:
                        item_items = item.strip().split('=')
                        if item_items[0] in model_infos:
                            if '.' in item_items[1] or 'e-' in item_items[1]:
                                loaded_model_params[item_items[0]] = float(item_items[1])
                            else:
                                loaded_model_params[item_items[0]] = int(item_items[1])
                elif task_type == 'regression':
                    model_infos = ['lr', 'wd', 'epochs', 'loss_valid', 'loss_train']
                    for item in model_name_items:
                        item_items = item.strip().split('=')
                        if item_items[0] in model_infos:
                            if '.' in item_items[1] or 'e-' in item_items[1]:
                                loaded_model_params[item_items[0]] = float(item_items[1])
                            else:
                                loaded_model_params[item_items[0]] = int(item_items[1])
            
                print(bcolors.OKGREEN + f"Load model: {load_model_status}" + bcolors.ENDC)
                print()
                print(f"Root Path: {bcolors.cyan_fg}{bcolors.underline}{loaded_model_info['path']}{bcolors.end_color}")
                print(f"Model Name: {bcolors.bold}{loaded_model_info['name']}.pt{bcolors.end_color}")
                print(f"Optimizer Name: {bcolors.bold}{optimizer_file}{bcolors.end_color}")
                print()
                print(bcolors.OKGREEN + f"loaded_lr: {loaded_model_params['lr']}"  + bcolors.ENDC)
                print(f"loaded_wd: {loaded_model_params['wd']}")
                print(bcolors.FAIL + f"loaded_epochs: {loaded_model_params['epochs']}" + bcolors.ENDC)
                return load_model_status, model, optimizer, loaded_model_info, loaded_model_params
            except Exception as e:
                print(bcolors.FAIL + f"Load model fail. {e}" + bcolors.ENDC)
                return e        

    @staticmethod
    def num_params(model, scale:int=1000000):
        nums = sum(p.numel() for p in model.parameters()) / scale
        return nums

    @staticmethod
    def create_result_dir(load_model_status, loaded_model_info, dir_name, model_info):
        current_date = datetime.now()
        model_date = (str(current_date.year) + "_" + str(current_date.month) + "_" +
                    str(current_date.day) + "_" + str(current_date.hour) + "_" +
                    str(current_date.minute))
        if load_model_status:
            source_code_path = loaded_model_info['path'][:-1][:loaded_model_info['path'][:].rfind("/")] + '/'
        else:
            source_code_path = file_handler_obj.make_dir(str(os.getcwd()), f'/{dir_name}')
        source_code_path = source_code_path.replace("\\", "/")
        print(f"Root Path: {bcolors.underline}{bcolors.cyan_fg}{source_code_path}{bcolors.end_color}")
        source_code_path = file_handler_obj.make_dir(source_code_path, str(str(model_date)) + ' ' + model_info)
        source_code_path = source_code_path.replace("\\", "/")
        print(f"Model Save Directory Path: {bcolors.underline}{bcolors.cyan_fg}{source_code_path}{bcolors.end_color}")
        return source_code_path

    @staticmethod
    def plot_train_progress(epoch_counter:int, train_hist:list, valid_hist:list, title:str, save_path:str):
        plt.plot(range(epoch_counter), train_hist, "r-", label="Train")
        plt.plot(range(epoch_counter), valid_hist, "b-", label="Validation")

        plt.xlabel("Epoch: " + str(epoch_counter))
        plt.ylabel(f"{title} "
                # + "T=" + str(f"{train_hist[-1]:.4}")
                # + " & "
                # + "V=" + str(f"{valid_hist[-1]:.4}")
        )
        x_spacing = 25
        y_spacing = 5
        x_minorLocator = MultipleLocator(x_spacing)
        y_minorLocator = MultipleLocator(y_spacing)
        plt.grid(visible=True, alpha=0.8, linewidth=1)
        plt.legend()
        ax = plt.gca()
        ax.yaxis.label.set_fontsize('large')
        ax.xaxis.label.set_fontsize('large')
        ax.yaxis.set_minor_locator(y_minorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        ax.grid(which = 'minor')
        plt.savefig(
            save_path
            + f"{title}"
            + f" epoch={len(valid_hist)}"
            + f" {title}_valid={valid_hist[-1]:.5f}"
            + f" {title}_train={train_hist[-1]:.5f}"
            + ".png"
        )
        pass
##########################################################################################
##########################################################################################
################ Model Save In Classification And Regression State #######################
    # Save Model Function Classification
    @staticmethod
    def save_model_classification(model, optimizer, epoch, loaded_epoch_counter, source_code_path, model_info,
                loss_train, best_loss_train, best_loss_train_saved_mode, best_loss_train_saved_optimizer,
                loss_train_hist, epoch_train_loss_hist_list, acc_train, best_acc_train, best_acc_train_saved_mode,
                best_acc_train_saved_optimizer, acc_train_hist, epoch_train_acc_hist_list,
                loss_valid, best_loss_valid, best_loss_valid_saved_mode, best_loss_valid_saved_optimizer,
                loss_valid_hist, epoch_valid_loss_hist_list, acc_valid, best_acc_valid, best_acc_valid_saved_mode,
                best_acc_valid_saved_optimizer, acc_valid_hist, epoch_valid_acc_hist_list,
                optimizer_info, highest_epoch_saved_mode, highest_epoch_saved_optimizer,
                f1_train_hist, f1_valid_hist, precision_train_hist, precision_valid_hist,
                recall_train_hist, recall_valid_hist):
        
        epochs_info = " epochs=" + str((epoch + 1) + loaded_epoch_counter)

        if loss_train < best_loss_train:
            if not(best_loss_train_saved_mode is None):
                os.remove(best_loss_train_saved_mode)
            # Save Best Loss Train Model
            best_loss_train_saved_mode = (str(source_code_path + "best_loss_train " + model_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
            torch.save(model, best_loss_train_saved_mode)

            # Save Best Loss Train Optimizer
            if not(best_loss_train_saved_optimizer is None):
                os.remove(best_loss_train_saved_optimizer)
            best_loss_train_saved_optimizer = (str(source_code_path + "best_loss_train " + optimizer_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
            torch.save(optimizer, best_loss_train_saved_optimizer)
            
            # Save Best Loss Train Hist Dta
            np.savetxt((source_code_path + 'loss_train_hist.txt'), loss_train_hist)
            np.savetxt((source_code_path + 'loss_valid_hist.txt'), loss_valid_hist)
            np.savetxt((source_code_path + 'acc_train_hist.txt'), acc_train_hist)
            np.savetxt((source_code_path + 'acc_valid_hist.txt'), acc_valid_hist)
            best_loss_train = loss_train
            print(bcolors.OKGREEN + f'Train: Loss = {loss_train:.5f}' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f'Train: Loss = {loss_train:.5f}' + bcolors.ENDC)

        if acc_train > best_acc_train:
            if not(best_acc_train_saved_mode is None):
                os.remove(best_acc_train_saved_mode)
            # Save Best Acc Train Model
            best_acc_train_saved_mode = (str(source_code_path + "best_acc_train " + model_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
            torch.save(model, best_acc_train_saved_mode)

            # Save Best Acc Train Optimizer
            if not(best_acc_train_saved_optimizer is None):
                os.remove(best_acc_train_saved_optimizer)
            best_acc_train_saved_optimizer = (str(source_code_path + "best_acc_train " + optimizer_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
            torch.save(optimizer, best_acc_train_saved_optimizer)
            
            # Save Best Acc Train Hist Dta
            np.savetxt((source_code_path + 'loss_train_hist.txt'), loss_train_hist)
            np.savetxt((source_code_path + 'loss_valid_hist.txt'), loss_valid_hist)
            np.savetxt((source_code_path + 'acc_train_hist.txt'), acc_train_hist)
            np.savetxt((source_code_path + 'acc_valid_hist.txt'), acc_valid_hist)
            best_acc_train = acc_train
            print(bcolors.OKGREEN + f'Train: Acc = {acc_train:.5f}' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f'Train: Acc = {acc_train:.5f}' + bcolors.ENDC)

        if loss_valid < best_loss_valid:
            if not(best_loss_valid_saved_mode is None):
                os.remove(best_loss_valid_saved_mode)
            best_loss_valid_saved_mode = (str(source_code_path + 'best_loss_valid ' + model_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
            torch.save(model, best_loss_valid_saved_mode)

            if not(best_loss_valid_saved_optimizer is None):
                os.remove(best_loss_valid_saved_optimizer)  
            best_loss_valid_saved_optimizer = (str(source_code_path + "best_loss_valid " + optimizer_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
            torch.save(optimizer, best_loss_valid_saved_optimizer)

            np.savetxt((source_code_path + 'loss_train_hist.txt'), loss_train_hist)
            np.savetxt((source_code_path + 'loss_valid_hist.txt'), loss_valid_hist)
            np.savetxt((source_code_path + 'acc_train_hist.txt'), acc_train_hist)
            np.savetxt((source_code_path + 'acc_valid_hist.txt'), acc_valid_hist)
            best_loss_valid = loss_valid
            print(bcolors.OKGREEN + f'Valid: Loss = {loss_valid:.5f}' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f'Valid: Loss = {loss_valid:.5f}' + bcolors.ENDC)

        if acc_valid > best_acc_valid:
            if not(best_acc_valid_saved_mode is None):
                os.remove(best_acc_valid_saved_mode)
            best_acc_valid_saved_mode = (str(source_code_path + 'best_acc_valid ' + model_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
            torch.save(model, best_acc_valid_saved_mode)

            if not(best_acc_valid_saved_optimizer is None):
                os.remove(best_acc_valid_saved_optimizer)  
            best_acc_valid_saved_optimizer = (str(source_code_path + "best_acc_valid " + optimizer_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
            torch.save(optimizer, best_acc_valid_saved_optimizer)

            np.savetxt((source_code_path + 'loss_train_hist.txt'), loss_train_hist)
            np.savetxt((source_code_path + 'loss_valid_hist.txt'), loss_valid_hist)
            np.savetxt((source_code_path + 'acc_train_hist.txt'), acc_train_hist)
            np.savetxt((source_code_path + 'acc_valid_hist.txt'), acc_valid_hist)
            best_acc_valid = acc_valid
            print(bcolors.OKGREEN + f'Valid: Acc = {acc_valid:.5f}' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f'Valid: Acc = {acc_valid:.5f}' + bcolors.ENDC)

        print()
        if not(highest_epoch_saved_mode is None):
            os.remove(highest_epoch_saved_mode)
        highest_epoch_saved_mode = (str(source_code_path + "highest_epoch_train " + model_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
        torch.save(model, highest_epoch_saved_mode)

        if not(highest_epoch_saved_optimizer is None):
            os.remove(highest_epoch_saved_optimizer)
        highest_epoch_saved_optimizer = (str(source_code_path + "highest_epoch_train " + optimizer_info + epochs_info +
                            ' acc_valid=' + str(f'{acc_valid:.5f}') +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' acc_train=' + str(f'{acc_train:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
        torch.save(optimizer, highest_epoch_saved_optimizer)

        np.savetxt((source_code_path + 'highest_loss_train_hist.txt'), loss_train_hist)
        np.savetxt((source_code_path + 'highest_loss_valid_hist.txt'), loss_valid_hist)

        np.savetxt((source_code_path + 'highest_acc_train_hist.txt'), acc_train_hist)
        np.savetxt((source_code_path + 'highest_acc_valid_hist.txt'), acc_valid_hist)

        np.savetxt((source_code_path + 'highest_f1_train_hist.txt'), f1_train_hist)
        np.savetxt((source_code_path + 'highest_f1_valid_hist.txt'), f1_valid_hist)

        np.savetxt((source_code_path + 'highest_precision_train_hist.txt'), precision_train_hist)
        np.savetxt((source_code_path + 'highest_precision_valid_hist.txt'), precision_valid_hist)

        np.savetxt((source_code_path + 'highest_recall_train_hist.txt'), recall_train_hist)
        np.savetxt((source_code_path + 'highest_recall_valid_hist.txt'), recall_valid_hist)

        np.savetxt((source_code_path + 'highest_epoch_loss_train_hist.txt'), epoch_train_loss_hist_list)
        np.savetxt((source_code_path + 'highest_epoch_loss_valid_hist.txt'), epoch_valid_loss_hist_list)
        np.savetxt((source_code_path + 'highest_epoch_acc_train_hist.txt'), epoch_train_acc_hist_list)
        np.savetxt((source_code_path + 'highest_epoch_acc_valid_hist.txt'), epoch_valid_acc_hist_list)
        
        return (best_loss_train, best_loss_train_saved_mode, best_loss_train_saved_optimizer,
        best_acc_train, best_acc_train_saved_mode, best_acc_train_saved_optimizer, 
        best_loss_valid, best_loss_valid_saved_mode, best_loss_valid_saved_optimizer,
        best_acc_valid, best_acc_valid_saved_mode, best_acc_valid_saved_optimizer,
        highest_epoch_saved_mode, highest_epoch_saved_optimizer)
    
    # Save Model Function Regression
    @staticmethod
    def save_model_regression(model, optimizer, epoch, loaded_epoch_counter, source_code_path, model_info,
                loss_train, best_loss_train, best_loss_train_saved_mode, best_loss_train_saved_optimizer,
                loss_train_hist, epoch_train_loss_hist_list,
                loss_valid, best_loss_valid, best_loss_valid_saved_mode, best_loss_valid_saved_optimizer,
                loss_valid_hist, epoch_valid_loss_hist_list,
                optimizer_info, highest_epoch_saved_mode, highest_epoch_saved_optimizer,
                mae_train_hist, mae_valid_hist, rmse_train_hist, rmse_valid_hist,
                r2_train_hist, r2_valid_hist):
        
        epochs_info = " epochs=" + str((epoch + 1) + loaded_epoch_counter)

        if loss_train < best_loss_train:
            if not(best_loss_train_saved_mode is None):
                os.remove(best_loss_train_saved_mode)
            # Save Best Loss Train Model
            best_loss_train_saved_mode = (str(source_code_path + "best_loss_train " + model_info + epochs_info +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
            torch.save(model, best_loss_train_saved_mode)

            # Save Best Loss Train Optimizer
            if not(best_loss_train_saved_optimizer is None):
                os.remove(best_loss_train_saved_optimizer)
            best_loss_train_saved_optimizer = (str(source_code_path + "best_loss_train " + optimizer_info + epochs_info +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
            torch.save(optimizer, best_loss_train_saved_optimizer)
            
            # Save Best Loss Train Hist Dta
            np.savetxt((source_code_path + 'loss_train_hist.txt'), loss_train_hist)
            np.savetxt((source_code_path + 'loss_valid_hist.txt'), loss_valid_hist)
            best_loss_train = loss_train
            print(bcolors.OKGREEN + f'Train: Loss = {loss_train:.5f}' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f'Train: Loss = {loss_train:.5f}' + bcolors.ENDC)

        if loss_valid < best_loss_valid:
            if not(best_loss_valid_saved_mode is None):
                os.remove(best_loss_valid_saved_mode)
            best_loss_valid_saved_mode = (str(source_code_path + 'best_loss_valid ' + model_info + epochs_info +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
            torch.save(model, best_loss_valid_saved_mode)

            if not(best_loss_valid_saved_optimizer is None):
                os.remove(best_loss_valid_saved_optimizer)  
            best_loss_valid_saved_optimizer = (str(source_code_path + "best_loss_valid " + optimizer_info + epochs_info +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
            torch.save(optimizer, best_loss_valid_saved_optimizer)

            np.savetxt((source_code_path + 'loss_train_hist.txt'), loss_train_hist)
            np.savetxt((source_code_path + 'loss_valid_hist.txt'), loss_valid_hist)
            best_loss_valid = loss_valid
            print(bcolors.OKGREEN + f'Valid: Loss = {loss_valid:.5f}' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f'Valid: Loss = {loss_valid:.5f}' + bcolors.ENDC)

        print()
        if not(highest_epoch_saved_mode is None):
            os.remove(highest_epoch_saved_mode)
        highest_epoch_saved_mode = (str(source_code_path + "highest_epoch_train " + model_info + epochs_info +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".pt"))
        torch.save(model, highest_epoch_saved_mode)

        if not(highest_epoch_saved_optimizer is None):
            os.remove(highest_epoch_saved_optimizer)
        highest_epoch_saved_optimizer = (str(source_code_path + "highest_epoch_train " + optimizer_info + epochs_info +
                            ' loss_valid=' + str(f'{loss_valid:.5f}') +
                            ' loss_train=' + str(f'{loss_train:.5f}') +
                            ".optim"))
        torch.save(optimizer, highest_epoch_saved_optimizer)

        np.savetxt((source_code_path + 'highest_loss_train_hist.txt'), loss_train_hist)
        np.savetxt((source_code_path + 'highest_loss_valid_hist.txt'), loss_valid_hist)

        np.savetxt((source_code_path + 'highest_epoch_loss_train_hist.txt'), epoch_train_loss_hist_list)
        np.savetxt((source_code_path + 'highest_epoch_loss_valid_hist.txt'), epoch_valid_loss_hist_list)

        np.savetxt((source_code_path + 'highest_mae_train_hist.txt'), mae_train_hist)
        np.savetxt((source_code_path + 'highest_mae_valid_hist.txt'), mae_valid_hist)

        np.savetxt((source_code_path + 'highest_rmse_train_hist.txt'), rmse_train_hist)
        np.savetxt((source_code_path + 'highest_rmse_valid_hist.txt'), rmse_valid_hist)

        np.savetxt((source_code_path + 'highest_r2_train_hist.txt'), r2_train_hist)
        np.savetxt((source_code_path + 'highest_r2_valid_hist.txt'), r2_valid_hist)

        return (best_loss_train, best_loss_train_saved_mode, best_loss_train_saved_optimizer,
                best_loss_valid, best_loss_valid_saved_mode, best_loss_valid_saved_optimizer,
                highest_epoch_saved_mode, highest_epoch_saved_optimizer)

##########################################################################################
##########################################################################################
################ Train And Validation Model In Classification State ######################

    # Training One Epoch Function Monoplex Classification
    @staticmethod
    def train_one_epoch_monoplex_classification(model, loader,
                loss_fn, optimizer,
                epoch: int = None, device: str = 'cuda',
                num_classes: int = None, task: str = 'multiclass'):
        # Set model to training mode
        model.train()
        loss_train = AverageMeter()
        current_epoch_losses = []

        # Initialize metrics
        f1_metric = F1Score(task=task, num_classes=num_classes).to(device)
        precision_metric = Precision(task=task, num_classes=num_classes).to(device)
        recall_metric = Recall(task=task, num_classes=num_classes).to(device)

        correct, total_samples = 0, 0
        
        # progress bar Initialization
        with tqdm(loader, unit="batch", colour='green') as tepoch:
            if epoch is not None:
                # Set description for the progress bar
                tepoch.set_description(f"Train Epoch {epoch}")
            # Iterate over batches
            for x_batch, y_batch in tepoch:

                # Move data to the specified device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                out = model(x_batch)
                
                # Compute loss
                loss_value = loss_fn(out, y_batch)

                # Backward pass
                loss_value.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                # Update parameters
                optimizer.step()
                
                # Update loss and accuracy trough meters
                loss_train.update(loss_value.item())
                current_epoch_losses.append(loss_value.item())

                pred = out.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total_samples += y_batch.size(0)

                # Update metrics
                f1_metric.update(pred, y_batch)
                precision_metric.update(pred, y_batch)
                recall_metric.update(pred, y_batch)
                
                # Update progress bar postfix
                tepoch.set_postfix(loss=f'{loss_train.avg:.5f}',
                                acc=f'{correct / total_samples:.5f}',
                                f1=f'{f1_metric.compute():.5f}',
                                prec=f'{precision_metric.compute():.5f}',
                                rec=f'{recall_metric.compute():.5f}')

        # return the average loss, accuracy, and metrics
        return model, loss_train.avg, (correct / total_samples), {
            'f1': f1_metric.compute().item(),
            'precision': precision_metric.compute().item(),
            'recall': recall_metric.compute().item()
        }, current_epoch_losses

    # Validation Function Monoplex Classification
    @staticmethod
    def validate_monoplex_classification(model, loader, loss_fn,
                epoch: int = None, device: str = 'cuda',
                num_classes: int = None, task: str = 'multiclass'):
        # Set model to evaluation mode
        model.eval()
        loss_valid = AverageMeter()
        current_epoch_losses = []

        # Initialize metrics
        f1_metric = F1Score(task=task, num_classes=num_classes).to(device)
        precision_metric = Precision(task=task, num_classes=num_classes).to(device)
        recall_metric = Recall(task=task, num_classes=num_classes).to(device)

        # Initialize correct and total samples
        correct, total_samples = 0, 0
        # with torch.no_grad() to disable gradient calculation
        with torch.no_grad():
            # progress bar Initialization
            with tqdm(loader, unit="batch", colour='blue') as tepoch:
                if epoch is not None:
                    # Set description for the progress bar
                    tepoch.set_description(f"Validation Epoch {epoch}")

                for x_batch, y_batch in tepoch:
                    # Move data to the specified device
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    # Forward pass
                    out = model(x_batch)
                    # Compute loss
                    loss_value = loss_fn(out, y_batch)

                    # Update loss and accuracy trough meters
                    loss_valid.update(loss_value.item())
                    current_epoch_losses.append(loss_value.item())

                    pred = out.argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                    total_samples += y_batch.size(0)

                    # Update metrics
                    f1_metric.update(pred, y_batch)
                    precision_metric.update(pred, y_batch)
                    recall_metric.update(pred, y_batch)
                    
                    # Update progress bar postfix
                    tepoch.set_postfix(loss=f'{loss_valid.avg:.5f}',
                                    acc=f'{correct / total_samples:.5f}',
                                    f1=f'{f1_metric.compute():.5f}',
                                    prec=f'{precision_metric.compute():.5f}',
                                    rec=f'{recall_metric.compute():.5f}')
        # return the average loss, accuracy, and metrics
        return loss_valid.avg, (correct / total_samples), {
            'f1': f1_metric.compute().item(),
            'precision': precision_metric.compute().item(),
            'recall': recall_metric.compute().item()
        }, current_epoch_losses

##########################################################################################
################ Train And Validation GNN Model In Classification State ##################
    # Training One Epoch Function GNN Model Monoplex Classification
    @staticmethod
    def train_one_epoch_monoplex_GNN_classification(model, loader,
                        loss_fn, optimizer,
                        epoch: int = None,
                        device: str = 'cuda',
                        num_classes: int = None):
        model.train()  # Set model to training mode
        loss_meter = AverageMeter()  # Tracks average loss
        all_losses = []  # List of batch-wise loss values

        # Initialize evaluation metrics
        accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        f1_metric = F1Score(task="multiclass", num_classes=num_classes).to(device)
        precision_metric = Precision(task="multiclass", num_classes=num_classes).to(device)
        recall_metric = Recall(task="multiclass", num_classes=num_classes).to(device)

        with tqdm(loader, unit="batch", colour='green') as tepoch:
            if epoch is not None:
                tepoch.set_description(f"Train Epoch {epoch}")

            for batch in tepoch:
                batch = batch.to(device)  # Move batch to device
                optimizer.zero_grad()     # Clear previous gradients

                # Forward pass
                out = model(batch.x, batch.edge_index)

                # Compute loss only on training samples
                loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()  # Backpropagation

                # Clip gradients to stabilize training
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()  # Update model weights

                # Track loss
                loss_meter.update(loss.item())
                all_losses.append(loss.item())

                # Compute predictions
                preds = out.argmax(dim=1)

                # Update metrics on training mask
                accuracy_metric.update(preds[batch.train_mask], batch.y[batch.train_mask])
                f1_metric.update(preds[batch.train_mask], batch.y[batch.train_mask])
                precision_metric.update(preds[batch.train_mask], batch.y[batch.train_mask])
                recall_metric.update(preds[batch.train_mask], batch.y[batch.train_mask])

                # Update progress bar
                tepoch.set_postfix(loss=f"{loss_meter.avg:.5f}",
                                acc=f"{accuracy_metric.compute():.5f}",
                                f1=f"{f1_metric.compute():.5f}",
                                prec=f"{precision_metric.compute():.5f}",
                                rec=f"{recall_metric.compute():.5f}")

        # Return final metrics
        return model, loss_meter.avg, {
            'accuracy': accuracy_metric.compute().item(),
            'f1': f1_metric.compute().item(),
            'precision': precision_metric.compute().item(),
            'recall': recall_metric.compute().item(),
        }, all_losses
    
    # Validation Function GNN Model Monoplex Classification
    @staticmethod
    def validate_monoplex_GNN_classification(model, loader, mask_type: str, loss_fn,
                epoch: int = None,
                device: str = 'cuda',
                num_classes: int = None):
            model.eval()  # Set model to evaluation mode
            loss_meter = AverageMeter()  # Tracks average loss
            all_losses = []  # List of individual batch losses

            # Initialize evaluation metrics
            accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
            f1_metric = F1Score(task="multiclass", num_classes=num_classes).to(device)
            precision_metric = Precision(task="multiclass", num_classes=num_classes).to(device)
            recall_metric = Recall(task="multiclass", num_classes=num_classes).to(device)

            with torch.no_grad():  # Disable gradient computation
                with tqdm(loader, unit="batch", colour='blue') as tepoch:
                    if epoch is not None:
                        tepoch.set_description(f"Validation Epoch {epoch}")

                    for batch in tepoch:
                        batch = batch.to(device)  # Move batch to device
                        mask = getattr(batch, mask_type)  # Get the mask (train/test/val)

                        # Forward pass
                        out = model(batch.x, batch.edge_index)

                        # Compute loss on masked data
                        loss = loss_fn(out[mask], batch.y[mask])
                        loss_meter.update(loss.item())
                        all_losses.append(loss.item())

                        # Compute predictions and update metrics
                        preds = out.argmax(dim=1)
                        accuracy_metric.update(preds[mask], batch.y[mask])
                        f1_metric.update(preds[mask], batch.y[mask])
                        precision_metric.update(preds[mask], batch.y[mask])
                        recall_metric.update(preds[mask], batch.y[mask])

                        # Update progress bar
                        tepoch.set_postfix(loss=f"{loss_meter.avg:.5f}",
                                        acc=f"{accuracy_metric.compute():.5f}",
                                        f1=f"{f1_metric.compute():.5f}",
                                        prec=f"{precision_metric.compute():.5f}",
                                        rec=f"{recall_metric.compute():.5f}")

            # Return all metrics
            return loss_meter.avg, {
                'accuracy': accuracy_metric.compute().item(),
                'f1': f1_metric.compute().item(),
                'precision': precision_metric.compute().item(),
                'recall': recall_metric.compute().item(),
            }, all_losses

##########################################################################################
##########################################################################################
################### Train And Validation Model In Regression State #######################
    # Training One Epoch Function Monoplex Regression
    @staticmethod
    def train_one_epoch_monoplex_regression(model, loader,
                                        loss_fn, optimizer,
                                        epoch: int = None, device: str = 'cuda'):
        model.train()  # Set model to training mode
        loss_train = AverageMeter()
        all_losses = []

        # Initialize regression metrics
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError(squared=True).to(device)  # Will use sqrt for RMSE
        r2_metric = R2Score().to(device)

        with tqdm(loader, unit="batch", colour='green') as tepoch:
            if epoch is not None:
                tepoch.set_description(f"Train Epoch {epoch + 1}")

            for x_batch, y_batch in tepoch:
                # Move data to the specified device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass and loss computation
                optimizer.zero_grad()
                out = model(x_batch).squeeze(dim=1)  # Output shape: [B]
                loss_value = loss_fn(out, y_batch)

                # Backpropagation and optimization
                loss_value.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                # Update loss and metrics
                loss_train.update(loss_value.item())
                all_losses.append(loss_value.item())
                mae_metric.update(out, y_batch)
                mse_metric.update(out, y_batch)
                r2_metric.update(out, y_batch)

                # Update tqdm display
                tepoch.set_postfix(dict(
                    loss=f'{loss_train.avg:.5f}',
                    MAE=f'{mae_metric.compute():.5f}',
                    RMSE=f'{mse_metric.compute().sqrt():.5f}',
                    R2=f'{r2_metric.compute():.5f}'
                ))

        # Compute final aggregated metrics
        final_mae = mae_metric.compute().item()
        final_rmse = mse_metric.compute().sqrt().item()
        final_r2 = r2_metric.compute().item()

        return  model, loss_train.avg, {
            'MAE': final_mae,
            'RMSE': final_rmse,
            'R2': final_r2
            }, all_losses
 
    # Validation Function Monoplex Regression
    @staticmethod
    def validate_monoplex_regression(model, loader, loss_fn,
                                    epoch: int = None, device: str = 'cuda'):
        model.eval()  # Set model to evaluation mode
        current_epoch_losses = []
        loss_valid = AverageMeter()

        # Initialize regression metrics
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError(squared=True).to(device)  # For RMSE
        r2_metric = R2Score().to(device)

        with torch.no_grad():  # No gradients needed
            with tqdm(loader, unit="batch", colour='blue') as tepoch:
                if epoch is not None:
                    tepoch.set_description(f"Test Epoch {epoch}")

                for x_batch, y_batch in tepoch:
                    # Move data to device
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Forward pass
                    out = model(x_batch).squeeze(dim=1)

                    # Compute loss
                    loss_value = loss_fn(out, y_batch)
                    loss_valid.update(loss_value.item())
                    current_epoch_losses.append(loss_value.item())

                    # Update metrics
                    mae_metric.update(out, y_batch)
                    mse_metric.update(out, y_batch)
                    r2_metric.update(out, y_batch)

                    # Update progress bar
                    tepoch.set_postfix(dict(
                        loss=f'{loss_valid.avg:.5f}',
                        MAE=f'{mae_metric.compute():.5f}',
                        RMSE=f'{mse_metric.compute().sqrt():.5f}',
                        R2=f'{r2_metric.compute():.5f}'
                    ))

        # Return all average metrics
        return loss_valid.avg, {
            'MAE': mae_metric.compute().item(),
            'RMSE': mse_metric.compute().sqrt().item(),
            'R2': r2_metric.compute().item()} , current_epoch_losses
##########################################################################################
################### Train And Validation GNN Model In Regression State #######################
    # Training One Epoch Function GNN Monoplex Regression
    @staticmethod
    def train_one_epoch_GNN_monoplex_regression(model, loader, loss_fn, optimizer,
                                    epoch: int = None,
                                    device: str = 'cuda'):
        model.train()
        loss_meter = AverageMeter()
        all_losses = []

        # Initialize metrics
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError(squared=True).to(device)  # RMSE needs sqrt
        r2_metric = R2Score().to(device)

        with tqdm(loader, unit="batch", colour='green') as tepoch:
            if epoch is not None:
                tepoch.set_description(f"Train Epoch {epoch}")

            for batch in tepoch:
                batch = batch.to(device)
                optimizer.zero_grad()

                out = model(batch.x, batch.edge_index).squeeze()
                loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                # Metric updates
                loss_meter.update(loss.item())
                all_losses.append(loss.item())
                mae_metric.update(out[batch.train_mask], batch.y[batch.train_mask])
                mse_metric.update(out[batch.train_mask], batch.y[batch.train_mask])
                r2_metric.update(out[batch.train_mask], batch.y[batch.train_mask])

                tepoch.set_postfix(dict(
                    loss=f"{loss_meter.avg:.5f}",
                    MAE=f"{mae_metric.compute():.5f}",
                    RMSE=f"{mse_metric.compute().sqrt():.5f}",
                    R2=f"{r2_metric.compute():.5f}"
                ))

        # Final metrics
        return {
            'model': model,
            'loss': loss_meter.avg,
            'MAE': mae_metric.compute().item(),
            'RMSE': mse_metric.compute().sqrt().item(),
            'R2': r2_metric.compute().item(),
            'batch_losses': all_losses
        }

    # Validation Function Monoplex Regression
    @staticmethod
    def validate_GNN_monoplex_regression(model, loader, mask_type: str,
                            loss_fn, epoch: int = None,
                            device: str = 'cuda'):
        model.eval()
        loss_meter = AverageMeter()
        all_losses = []

        # Metrics
        mae_metric = MeanAbsoluteError().to(device)
        mse_metric = MeanSquaredError(squared=True).to(device)
        r2_metric = R2Score().to(device)

        with torch.no_grad():
            with tqdm(loader, unit="batch", colour='blue') as tepoch:
                if epoch is not None:
                    tepoch.set_description(f"Validation Epoch {epoch}")

                for batch in tepoch:
                    batch = batch.to(device)
                    mask = getattr(batch, mask_type)

                    out = model(batch.x, batch.edge_index).squeeze()
                    loss = loss_fn(out[mask], batch.y[mask])

                    # Update loss and metrics
                    loss_meter.update(loss.item())
                    all_losses.append(loss.item())
                    mae_metric.update(out[mask], batch.y[mask])
                    mse_metric.update(out[mask], batch.y[mask])
                    r2_metric.update(out[mask], batch.y[mask])

                    tepoch.set_postfix(dict(
                        loss=f"{loss_meter.avg:.5f}",
                        MAE=f"{mae_metric.compute():.5f}",
                        RMSE=f"{mse_metric.compute().sqrt():.5f}",
                        R2=f"{r2_metric.compute():.5f}"
                    ))

        return {
            'loss': loss_meter.avg,
            'MAE': mae_metric.compute().item(),
            'RMSE': mse_metric.compute().sqrt().item(),
            'R2': r2_metric.compute().item(),
            'batch_losses': all_losses
        }

##########################################################################################
    @staticmethod
    def small_gride(create_model, train_loader:DataLoader, loss_fn:nn.HuberLoss, epoch_cun:int, device:str='cpu'):
        best_lr = 0.0001
        best_wd = 1e-5
        delta = -1
        num_epochs = epoch_cun
        for lr in [0.01, 0.009, 0.007, 0.005, 0.003, 0.001, 0.0009, 0.0007, 0.0005, 0.0003, 0.0001]:
            for wd in [1e-4, 1e-5, 0.]:
                model = create_model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                print(f'LR={lr}, WD={wd}')
                start_loss = torch.inf
                end_loss = torch.inf
                for epoch in range(num_epochs):
                    model, loss, _ = Neural_Network_Functions.train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
                if epoch == 0:
                    start_loss = loss
                else:
                    end_loss = loss
                if (start_loss - end_loss) > delta:
                    delta = start_loss - end_loss
                    best_lr = lr
                    best_wd = wd
        return best_lr, best_wd
    
    pass