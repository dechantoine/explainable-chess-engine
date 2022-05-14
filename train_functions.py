import logging
import os
import json

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split


def data_preparation(X, y, frac, batch_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=frac, random_state=seed)

    traindataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train,y_train), batch_size=batch_size,
                                                  shuffle=False, num_workers=0)
    testdataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test,y_test), batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
    
    return X_train, X_test, y_train, y_test, traindataloader, testdataloader


def load_running_loss_evol(path_folder, resume_at_epoch=0):
    if resume_at_epoch==0:
        val_results = dict()
        with open(path_folder + "/validation_results.json", 'w') as fp:
            json.dump(val_results, fp, indent=4)
        running_loss_evol = pd.DataFrame(columns=["epoch", "batch", "mean loss", "learning rate", "duration"])
    else:
        running_loss_evol = pd.read_csv(path_folder + "/loss_evol.csv", index_col=0)
        running_loss_evol = running_loss_evol[running_loss_evol["epoch"]<resume_at_epoch]
    
    return running_loss_evol


def validation(model, testloader, device):
    
    logger = logging.getLogger("training_logger")
    logger.info("                                     /// Validation statistics ///                                     ")

    model.eval()
    val_targets=[]
    val_outputs=[]
    with torch.no_grad():
        for i, batch in enumerate(testloader, 0):
            input_boards = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs_nn = model(input_boards)
            val_targets.extend(labels.cpu().detach().numpy())
            val_outputs.extend(outputs_nn.cpu().detach().numpy())
            
    val_outputs = np.array(val_outputs).flatten()
    
            
    results_distribution = pd.DataFrame(index=["[" + str(c[0]) + ", " + str(c[1]) + "]" 
                                               for c in np.sort(np.reshape(np.array([np.round(np.arange(-2.0,2,0.1),2),
                                                                                     np.round(np.arange(-1.9,2.1,0.1),2)]),
                                                                           (40,2)),
                                                                axis=0)],
                           columns=["targets distribution", "predictions distribution",
                                    "errors distribution"])
    
    errors = (np.array(val_targets) - val_outputs).flatten()
    squared_errors = ((np.array(val_targets) - val_outputs)**2).flatten()
    
    results_distribution.iloc[10:30,0] = np.histogram(val_targets,
                                                      bins=np.round(np.arange(-1.0,1.1,0.1),2))[0]/len(val_targets)
    results_distribution.iloc[10:30,1] = np.histogram(val_outputs,
                                                      bins=np.round(np.arange(-1.0,1.1,0.1),2))[0]/len(val_outputs)
    results_distribution.iloc[:,2] = np.histogram(errors, bins=np.round(np.arange(-2.0,2.1,0.1),2))[0]/len(errors)
    
    results_stats = pd.DataFrame(columns=["targets", "predictions","errors", "absolute_errors"],
                                 index=["mean", "std", "count", "min",
                                        "10%", "20%", "30%", "40%", "50%",
                                        "60%", "70%", "80%", "90%", "max"])
    results_stats.loc["mean", "targets"] = np.mean(val_targets)
    results_stats.loc["mean", "predictions"] = np.mean(val_outputs)
    results_stats.loc["mean", "errors"] = np.mean(errors)
    results_stats.loc["mean", "absolute_errors"] = np.mean(abs(errors))
    results_stats.loc["mean", "squared_errors"] = np.mean(squared_errors)
    results_stats.loc["std", "targets"] = np.std(val_targets)
    results_stats.loc["std", "predictions"] = np.std(val_outputs)
    results_stats.loc["std", "errors"] = np.std(errors)
    results_stats.loc["std", "absolute_errors"] = np.std(abs(errors))
    results_stats.loc["std", "squared_errors"] = np.std(squared_errors)
    results_stats.loc["count", "targets"] = int(len(val_targets))
    results_stats.loc["count", "predictions"] = int(len(val_outputs))
    results_stats.loc["count", "errors"] = int(len(errors))
    results_stats.loc["count", "absolute_errors"] = int(len(errors))
    results_stats.loc["count", "squared_errors"] = int(len(errors))
    results_stats.iloc[3:, :] = pd.DataFrame([val_targets,
                                              val_outputs,
                                              errors,
                                              abs(errors),
                                              squared_errors]).T.quantile(np.arange(0,1.1,0.1))
    
    return results_distribution, results_stats