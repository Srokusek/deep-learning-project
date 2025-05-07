import torch
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data.dataset import CLASS_NAMES
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def eval(model, dataset, categories, batch_size, device, label="", return_ordered=False):
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

    #create the text features (embeddings), 
    #can be done once and then assigned to same categories
    text_features = model.encode_text(text_inputs) #(num_classes, feature_dims)
    text_features /= text_features.norm(dim=1, keepdim=True) #normalize text features (standard paractice with CLIP)

    #initiate the torch dataloader which, efficiently loads the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=label):
        #load tartget class index and convert to tensor of integers (.long() in pytorch)
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long() #(batch_size,)

        image = image.to(device)
        target = target.to(device)

        #get the image features for all images in batch
        image_features = model.encode_image(image) #(batch_size, feature_dims)
        image_features /= image_features.norm(dim=1, keepdim=True)

        #calculate the predicted class
        logits = (image_features @ text_features.T)
        predicted_class = logits.argmax(dim=1) #(batch_size, feature_dims) @ (feature_dims, num_classes) -> (batch_size, num_classes)
                                               #(batch_size, num_classes).argmax(dim=1) -> (batch_size)
        
        if return_ordered:
            _, indices = torch.sort(logits, dim=1, descending=True) #returns the logits ordered from high to low, along with the indices tensor
            correct_positions = (indices == target.unsqueeze(1)).nonzero(as_tuple=True) #1 where true, 0 otherwise, unsqueeze truns it from (128) to (128,1) -> can broadcoast across 
            predictions_order = correct_positions[1].to("cpu") #get index of correct prediction = order at which the index was predicted
            correct_positions_list.append(predictions_order)
            target_list.append(target)

        #need to use .item() because we obtain tensor(x)
        correct_predictions += (predicted_class==target).sum().item()

    accuracy = correct_predictions/len(dataset)

    if return_ordered:
        correct_positions_list = torch.cat(correct_positions_list).cpu().numpy()
        target_list = torch.cat(target_list).cpu().numpy()
        return correct_positions_list, target_list
    return accuracy

#reference from https://github.com/openai/CLIP readme
@torch.no_grad()
def get_features(model, dataset, batch_size, label):
    all_features = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for image, label in tqdm(dataloader, label):
        #get the features for all images in the dataset
        features = model.encode_image(image.to(device))

        all_features.append(features)
        all_labels.append(label)
    
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def linear_probe_evaluation(model, train_set, test_set, batch_size):
    train_features, train_labels = get_features(model, train_set, batch_size=batch_size, label="🖼️Extracting features of training set")
    test_features, test_labels = get_features(model, test_set, batch_size=batch_size, label="🖼️Extracting features of test set")

    classifier = LogisticRegression(random_state=44, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels==predictions).astype("float")) * 100

    return accuracy