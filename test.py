import os

import numpy as np
import torch
import torch.nn.functional as F

import params
from dataloader import get_cleaned_features
from models.iab_models import SimpleRNN
from tools.ml_utils import return_perf_metric
from tools.utils import save_json


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def test_model(model,
               test_dataloader,
               criterion,
               device=torch.device('cpu')):
    # Load validation data at once
    test_data, test_target = next(iter(test_dataloader))
    test_losses = []

    # Evaluate model on data
    model.eval()
    test_output = model(test_data.to(device))  # .permute(1, 0, 2)
    test_loss = criterion(test_output, test_target.to(device).squeeze(-1))  # .double().view(-1, 1)
    test_losses.append(test_loss.item())

    test_predictions = np.array(test_output.cpu().detach())
    # compute metrics
    test_perf = return_perf_metric(np.around(test_target.numpy()).squeeze(-1),
                                   np.around(test_predictions),
                                   ["accuracy", "f1", "recall", "precision", "auroc"])
    # validation_mse_score = mean_squared_error(validation_target.numpy(), validation_output.cpu().detach().numpy())
    return model, test_perf


def main(param, prefix=""):
    feat_names = sorted(param.features, reverse=False)
    feat_prefix = ""
    for f in feat_names: feat_prefix += f[0]
    model_path = f"Resultats/{prefix}_{feat_prefix}_{param.sequence_time}_lr{param.lr}_epoch{param.epochs}_hdim{param.hidden_dims}.pt"
    # Seed the random numbers generator for reproducibility
    torch.manual_seed(0)

    # Limit the number of threads created to parallelize CPU operations to 1
    torch.set_num_threads(1)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    train_dataset, validation_dataset, input_dim, output_dim = get_cleaned_features(param.dataset_path,
                                                                                    param.test_video,
                                                                                    param.openface_feat_names,
                                                                                    param.blocksize,
                                                                                    param.sequence_time,
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
    # Load the state dict
    state_dict = torch.load(model_path)  # Update the model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.double()

    criterion = F.binary_cross_entropy  # F.cross_entropy

    model, perf = test_model(model, train_dataloader, validation_dataloader, criterion)

    save_json(perf,
              f"Results/test/{param.prefix}_{param.feat_prefix}_{param.sequence_time}_lr{param.lr}_epoch{param.epochs}_hdim{param.hidden_dims}.json")
    # show fold result
    for k in perf.keys():
        print(f"{k}:{perf[k]}")
    return perf


if __name__ == "__main__":
    main(params)
