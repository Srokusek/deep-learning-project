import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from tqdm import tqdm
import copy
import time

from src.models.clip_wrapper import load_clip_model
from src.data.dataset import CLASS_NAMES


#Simplified version of original code https://github.com/dosowiechi/CLIPArTT/blob/main/models/tent.py

class NewCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, self.preprocess, _ = load_clip_model()
        self.logit_scale = self.clip.logit_scale

    def forward(self, x, labels):
        image_features = self.encode_image(x)
        text_features = self.encode_text(labels)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features

    def encode_image(self, image):
        return self.clip.encode_image(image)
    
    def encode_text(self, text):
        return self.clip.encode_text(text)

#function for creating pseudoprompts
def getprompt(K, c, categories):
    for k in range(K):
        if k == 0:
            text_prompt = f"a photo of a " + categories[c[k]] #label for the nearest neighbor
        else:
            text_prompt = text_prompt + " or " + categories[c[k]] #iteratively add subsequent prompts
    return text_prompt

#copy the model and optimizer before reseting
def copy_model_and_optimizer(model, optimizer):
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

#load saved model state
def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

#define the cross entropy loss to allow for easy experimentation
#in this caase targets can be tensors, not just class labels
def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

@torch.enable_grad() #need to make sure grad is enabled even when evaluating
def forward_and_adapt(x, text_x, categories, device, model, optimizer, K=3, target_method=1):
    """
    forward and adapt model on batch of data
    """

    #get the embeddings
    with torch.no_grad():
        image_features = model.encode_image(x)
        text_features = model.encode_text(text_x)

    #normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    #print(text_features.shape)

    #compute the similarity between the image embeddings and text embeddings (classic CLIP)
    similarity = (100.0 * image_features @text_features.T).softmax(dim=-1) #converting to probabilities mainly just for readability
    values, pred = similarity.topk(K, 1, True, True) #torch has a topk method! (K, dim=1, largest=True, sorted=True)
    
    pred_inputs = torch.cat([clip.tokenize(getprompt(K, c, categories)) for c in pred]).to(device) #create and tokenize the pseudolabel for top k categories c, 
                                                                                              #then join into a single tensor
    
    #calculate pseudolabel features
    logits, image_features, text_features = model(x, pred_inputs) #in this case uses and adapted clip which returns the features along with logits

    #print(image_features.shape)
    images_similarity = image_features @ image_features.t() #cross similarity between image features (in batch???)
    texts_similarity = text_features @ text_features.t() #cross similarity between text features (in batch???)

    if target_method == 1: #mean between cross similarities
        targets = F.softmax(
            ((images_similarity + texts_similarity) / 2)/0.01, dim=-1
        )
    elif target_method == 2: #only look at image cross similarity
        targets = F.softmax(
            ((images_similarity) / 2)/0.01, dim=-1
        )
    elif target_method == 3: #only look at text cross similarity
        targets = F.softmax(
            ((texts_similarity) / 2)/0.01, dim=-1
        )

    loss = cross_entropy(logits.t(), targets.t(), reduction="mean") #we use the cross similarities as soft label targets, remember we do not have the actual labels!
    #this loss helps to minimize the representation difference

    #print(images_similarity.shape)

    #make a optimizer step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

class Tent(nn.Module):
    def __init__(self, 
                 model, 
                 optimizer, 
                 steps=1,
                 episodic = False #defines whether or not the model resets at every step
        ):
        super().__init__()

        #save the parameters
        self.model = model
        self.preprocess = model.preprocess
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires steps >= 1"
        self.episodic = episodic

        #save the model if we plan to reset it
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, text_x, categories, device, K=5, target_method=1, affinity="knn", force_symmetry=True):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            forward_and_adapt(x, text_x, categories, device, self.model, self.optimizer, K=K, target_method=target_method)

        #make adapted clip prediction
        logits, image_features, text_features = self.model(x, text_x)

        return logits
    
    def reset(self):
        #if resetting is not done, the model finetuning decrases performance, especially on zero shot data
        #load the original parameters
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

def configure_model(model): #simplified version of applying tent to the model
    
    model.requires_grad_(False) #freeze the model
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True) #we fine tune on the layer norms
    return model

def clipartt_eval(model, dataset, categories, batch_size, device, label="", return_ordered=False):
    model.eval()
    if return_ordered:
        correct_positions_list = []
        target_list =[]

    #create contigous dictionary in the form category: index
    #helps to map categories to indices
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    #create and tokenize a text for each of the categories, 
    #enables using the text model
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(device)

    #initiate the torch dataloader which, efficiently loads the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=label):
        #load tartget class index and convert to tensor of integers (.long() in pytorch)
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long() #(batch_size,)

        image = image.to(device)
        target = target.to(device)

        logits = model(image, text_inputs, CLASS_NAMES, device)
        predicted_class = logits.argmax(dim=1)
        correct_predictions += (predicted_class==target).sum().item()
    accuracy = correct_predictions/len(dataset)

    return accuracy