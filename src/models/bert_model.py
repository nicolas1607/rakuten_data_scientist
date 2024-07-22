
import torch
import random
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForPreTraining
from transformers import get_linear_schedule_with_warmup

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def evaluate(model, device, dataloader_test):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_test:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_test) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def evaluateTest(model, device, dataloader_test):
    
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_test:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_test)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def result_generation(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

        P_c = precision_score(labels_flat, preds_flat, average=None, labels=[label])[0]
        R_c = recall_score(labels_flat, preds_flat, average=None, labels=[label])[0]
        F1_c = f1_score(labels_flat, preds_flat, average=None, labels=[label])[0]

        print(f"precision:\t{P_c:.4f}")
        print(f"recall:\t\t{R_c:.4f}")
        print(f"F1 score:\t{F1_c:.4f}")
        print()

    P = precision_score(labels_flat, preds_flat, average='micro')
    R = recall_score(labels_flat, preds_flat, average='micro')
    F1 = f1_score(labels_flat, preds_flat, average='micro')

    print("=*= global =*=")
    print(f"precision:\t{P:.4f}")
    print(f"recall:\t\t{R:.4f}")
    print(f"F1 score:\t{F1:.4f}")
    print()

def modele_bert(X_train, X_test, y_train, y_test):

    print("Modèlisation BERT\n")

    # Transformer les labels en entiers de 0 à X
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Tokenisation des données
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_data_train = tokenizer.batch_encode_plus(
        X_train.values.tolist(),
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding='longest',
        truncation=True, 
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        X_test.values.tolist(),
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding='longest',
        truncation=True, 
        return_tensors='pt'
    )

    # Préparation des données
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(list(y_train))

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(list(y_test))

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Création du modèle
    model = BertForPreTraining.from_pretrained(
        "bert-base-uncased", num_labels=27, output_attentions=False, output_hidden_states=False, class_weights=class_weights
    )
    
    # Création des dataloader
    batch_size = 4
    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)
    
    # Paramètres du modèle
    epochs = 1
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
    
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in tqdm(range(1, epochs+1)):
    
        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
                    
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(model, device, dataloader_test)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

    pickle.dump(model, open("models/bert.pkl", "wb"))

    loss_val_avg, predictions, true_vals = evaluateTest(model, device, dataloader_test)

    print("=*= Résultats =*=\n")
    print(f"Validation loss: {loss_val_avg}")
    result_generation(predictions, true_vals)