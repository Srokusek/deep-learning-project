import torch

#LoRA layer
class LoraLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__() #make sure nn.Module is initialized properly
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev) #randomly initiate a (in_dim, rank) tensor
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)) #initiate a zeros (rank, out_dim) tensor
        self.alpha = alpha #scales the size of the influence on the original tensor

    def forward(self, x):
        if self.A.dtype != x.dtype:
            self.A.data = self.A.data.to(device=x.device, dtype=x.dtype)
            self.B.data = self.B.data.to(device=x.device, dtype=x.dtype)
        return self.alpha * (x @ self.A @ self.B)
    

#combo layer of linear layer + LoRA
class LinearPlusLora(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        #apply lora to any given linear layer
        self.linear = linear
        self.lora = LoraLayer(
            in_dim=linear.in_features,
            out_dim=linear.out_features, 
            rank=rank, 
            alpha=alpha
        )

        #expose parameters of linear layer for attention blocks
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias if hasattr(linear, 'bias') else None

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    
def add_lora_to_model(model, rank, alpha):
    target_modules = [
        "mlp.c_fc",
        "mlp.c_proj"
    ]

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(t in name for t in target_modules):
            parent_name, child_name = name.rsplit(".", 1)
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            
            lora_layer = LinearPlusLora(module, rank, alpha)
            setattr(parent, child_name, lora_layer)