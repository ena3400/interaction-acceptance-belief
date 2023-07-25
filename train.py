import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import params
from dataloader import get_cleaned_features
# import models
from models.iab_models import SimpleRNN
from tools.ml_utils import return_perf_metric
from tools.utils import save_json, check_and_create_folder


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model,
                train_dataloader,
                validation_dataloader,
                criterion,
                optimizer,
                epochs=50,
                device=torch.device('cpu')):
    """
    :param model: Model to train
    :param train_dataloader: Loads training data
    :param validation_dataloader: Loads validation data we use to monitor training & save the best model
    :param criterion: Loss function used for training
    :param optimizer: Optimizer used for training (Adam is usually the preferred choice)
    :param trained_model_name: Trained model is saved under this name
    :param epochs: Number of training epochs
    :param device: CPU or GPU
    """

    # Load validation data at once
    validation_data, validation_target = next(iter(validation_dataloader))

    # Lists to save train & test losses
    train_losses = []
    validation_losses = []

    # Initialize best validation F1 score
    best_validation_score = 0.
    best_val_perf = {"f1": 0, "accuracy": 0, "recall": 0, "precision": 0, "auroc": 0}
    best_state_dict = copy.deepcopy(model.state_dict())

    # Training loop
    for epoch in range(epochs):
        epoch_labels = []
        epoch_predictions = []
        batch_losses = []
        # Set train mode
        model.train()

        # Load training data batch
        for data, target in train_dataloader:
            # Clear gradients
            model.zero_grad()

            # Permute data axes so that it can be processed by the RNN, then run forward pass
            output = model(data.to(device))  # .permute(1, 0, 2)

            # print(output.shape)
            # print(target.to(device).shape)
            # Compute loss, gradients, and update model parameters
            loss = criterion(output, target.to(device))  # .double().view(-1, 1)

            # track prediction, loss over epochs
            prob = output.data.cpu().detach().numpy()
            label = target.data.cpu().numpy()
            epoch_predictions += prob.tolist()
            epoch_labels += label.tolist()
            batch_losses.append(loss.data.cpu().numpy())

            # update
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()

        # compute loss of epoch
        epoch_loss = np.array(batch_losses).mean()
        train_losses.append(epoch_loss)

        epoch_predictions = np.array(epoch_predictions)
        # compute training metrics
        train_perf = return_perf_metric(np.around(np.array(epoch_labels).squeeze()),
                                        np.around(epoch_predictions),
                                        ["accuracy", "f1", "recall", "precision", "auroc"])

        # Evaluate model on validation data
        model.eval()
        validation_output = model(validation_data.to(device))  # .permute(1, 0, 2)
        validation_loss = criterion(validation_output,
                                    validation_target.to(device).squeeze(-1))  # .double().view(-1, 1)
        validation_losses.append(validation_loss.item())

        val_predictions = np.array(validation_output.cpu().detach())
        # compute val metrics
        val_perf = return_perf_metric(np.around(validation_target.numpy()).squeeze(-1),
                                      np.around(val_predictions),
                                      ["accuracy", "f1", "recall", "precision", "auroc"])
        # validation_mse_score = mean_squared_error(validation_target.numpy(), validation_output.cpu().detach().numpy())

        print(
            'Epoch: %0.f %0.7f |  %0.4f  %0.4f %0.4f %0.4f | %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f' % (
                int(epoch), get_lr(optimizer),
                epoch_loss, train_perf["accuracy"], train_perf["f1"], train_perf["auroc"],
                validation_loss, val_perf["accuracy"], val_perf["f1"], val_perf["recall"], val_perf["precision"],
                val_perf["auroc"]))

        if val_perf["f1"] > best_validation_score:
            best_validation_score = val_perf["f1"]
            best_val_perf = val_perf
            best_state_dict = copy.deepcopy(model.state_dict())
            print("Model saved at epoch {:}. Validation f1 score: {:.2f}".format(epoch + 1,
                                                                                 best_validation_score))
        print("--------------------------------------------------")
    print("End of training")
    return best_state_dict, best_val_perf


def main(param,
         prefix):
    # Seed the random numbers generator for reproducibility
    torch.manual_seed(0)

    # Limit the number of threads created to parallelize CPU operations to 1
    torch.set_num_threads(1)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    feat_names = sorted(param.features, reverse=False)
    feat_prefix = ""
    for f in feat_names: feat_prefix += f[0]

    all_perf = []
    best_perf = 0
    top_perf = "f1"
    for fold, video_val in enumerate(param.Y_cross_val):
        print(f"New fold")
        train_dataset, validation_dataset, input_dim, output_dim = get_cleaned_features(param.dataset_path, video_val,
                                                                                        param.openface_feat_names,
                                                                                        param.blocksize, param.sequence_time,
                                                                                        feat_names,
                                                                                        resampling=param.resampling)

        gpu_available = torch.cuda.is_available()
        device = torch.device('cuda:0') if gpu_available else torch.device('cpu')

        # Create train & validation data loaders
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset))

        # Set model parameters for the training
        model = SimpleRNN(input_dim, param.hidden_dims, output_dim,
                          attention_context=False,
                          bidirectional=True)  # torch.nn.GRU(input_dim, args.hidden_dims, num_layers=1)
        model.to(device)
        model.double()

        criterion = F.binary_cross_entropy  # F.cross_entropy
        # criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)

        best_state_dict, perf = train_model(model, train_dataloader, validation_dataloader, criterion, optimizer,
                                            param.epochs,
                                            device)
        all_perf.append(perf)
        if perf[top_perf] > best_perf:
            best_perf = perf[top_perf]
            print(f"New best perf {top_perf}:{perf[top_perf]}, old one {best_perf}")
            torch.save(best_state_dict,
                       f"Results/models/{prefix}_{feat_prefix}_{param.sequence_time}_lr{param.lr}_epoch{param.epochs}_hdim{param.hidden_dims}.pt")
            save_json(perf,
                      f"Results/models/{prefix}_{feat_prefix}_{param.sequence_time}_lr{param.lr}_epoch{param.epochs}_hdim{param.hidden_dims}.json")
    # show fold result
    print(f"Average on all folds:")
    for k in all_perf[0].keys():
        metric = 0
        for perf in all_perf:
            metric += perf[k]
        print(f"{k}:{metric / len(all_perf)}")
    return all_perf


def run_one_kfold(param, prefix=""):
    perf = main(param,
                prefix)
    check_and_create_folder("Results/folds")
    check_and_create_folder("Results/models")
    feat_names = sorted(param.features, reverse=False)
    feat_prefix = ""
    for f in feat_names: feat_prefix += f[0]
    save_json(perf,
              f"Results/folds/{prefix}_{feat_prefix}_{param.sequence_time}_lr{param.lr}_epoch{param.epochs}_hdim{param.hidden_dims}.json")


if __name__ == "__main__":
    run_one_kfold(params)
