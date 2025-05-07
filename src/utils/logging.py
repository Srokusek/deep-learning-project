import wandb
import gc
import torch

def inspect_model_training(model, epoch, train_loss=None, val_accuracy=None):
    """Log information about training process"""
    grad_norms = []
    param_norms = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = param.grad
            grad_status = "Has grad" if grad is not None else "No grad"
            grad_norm = grad.norm().item() if grad is not None else 0
            param_norm = param.norm().item()

            #log layer specific details
            grad_norms.append(grad_norm)
            param_norms.append(param_norm)
    
    #log global details
    global_logs = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuarcy": val_accuracy
    }

    #log to wandb
    wandb.log({**global_logs,
               "param_norms_hist": wandb.Histogram(param_norms),
               "grad_norms_histogram": wandb.Histogram(grad_norms)
               })
    

def inspect_trainable_parameters(model):
    """Inspect number of learnable parameters"""
    trainable_params = 0
    all_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    param_logs = {
        "params_total": all_params,
        "params_trainable": trainable_params,
        "percentage_trainable": trainable_params / all_params * 100
    }
    
    wandb.log(param_logs)

def dump_cuda_cache():
    if "model" in globals(): #check if model is a global variable
        global model
        model = model.cpu() #move the model to cpu

    torch.cuda.empty_cache()
    gc.collect()