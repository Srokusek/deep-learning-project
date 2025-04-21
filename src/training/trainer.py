import torch
from tqdm import tqdm
from src.data.dataset import CLASS_NAMES
import clip

def get_optimizer(model, learning_rate, weight_decay, momentum):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    return optimizer

def get_cost_function():
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function

def training_step(model, dataset, categories, batch_size, optimizer, cost_function, device):
    total_loss = 0
    
    model = model.to(device)


    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs) #(num_classes, feature_dims)
        text_features = text_features / text_features.norm(dim=1, keepdim=True) #normalize text features (standard paractice with CLIP)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    with tqdm(dataloader, desc='Training') as pbar:
        for image, target in pbar:
            #reset gradients
            optimizer.zero_grad()

            target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long()

            #transfer relevant data to gpu
            image = image.to(device)
            target = target.to(device)

            #get the image features
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            #predict the class by explicit matrix multiplication
            logits = image_features @ text_features.T

            #calculate the loss
            loss = cost_function(logits, target)

            #backprop
            loss.backward()
            optimizer.step()

            #get loss and prediction
            total_loss += loss.item()

            #update progress bar
            pbar.set_postfix(train_loss=loss.item())

    return total_loss/len(dataloader)