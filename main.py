import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import logging

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *

from timo_utils import refine_prompts, refine_images

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move tensors to same device and correct dtype
    val_features = val_features.to(device).float()
    cache_keys = cache_keys.to(device).float()
    cache_values = cache_values.to(device).float()
    clip_weights = clip_weights.to(device).float()
    val_labels = val_labels.to(device)
    
    # Log shapes for debugging
    print(f"Debug - run_tip_adapter shapes: val_features: {val_features.shape}, cache_keys: {cache_keys.shape}, cache_values: {cache_values.shape}, clip_weights: {clip_weights.shape}")
    
    # Enforce float32 dtype
    
    # Shape assertions
    feature_dim = val_features.shape[1]
    assert cache_keys.shape[1] == feature_dim, f"Feature dimension mismatch: val_features {val_features.shape}, cache_keys {cache_keys.shape}"
    assert clip_weights.shape[0] == feature_dim, f"Feature dimension mismatch: val_features {val_features.shape}, clip_weights {clip_weights.shape}"

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    # Matrix multiplication with proper transposition
    affinity = val_features @ cache_keys.t()
    
    # Shape assertion after matrix multiplication
    assert affinity.shape[0] == val_features.shape[0], f"Affinity row count mismatch: expected {val_features.shape[0]}, got {affinity.shape[0]}"
    assert affinity.shape[1] == cache_keys.shape[0], f"Affinity column count mismatch: expected {cache_keys.shape[0]}, got {affinity.shape[1]}"
    
    # Convert both tensors to the same dtype (float32) before matrix multiplication
    exp_term = ((-1) * (beta - beta * affinity)).exp().float()
    cache_values_float = cache_values.float()
    
    # Shape assertion for cache_logits calculation
    assert exp_term.shape[1] == cache_values_float.shape[0], f"Shape mismatch: exp_term {exp_term.shape}, cache_values {cache_values_float.shape}"
    
    cache_logits = exp_term @ cache_values_float
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")
    
    # Log shapes for debugging
    print(f"Debug - run_tip_adapter test shapes: test_features: {test_features.shape}, cache_keys: {cache_keys.shape}")
    
    # Enforce float32 dtype
    test_features = test_features.to(device).float()
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    test_labels = test_labels.to(device)
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    # Matrix multiplication with proper transposition
    affinity = test_features @ cache_keys.t()
    
    # Shape assertion after matrix multiplication
    assert affinity.shape[0] == test_features.shape[0], f"Affinity row count mismatch: expected {test_features.shape[0]}, got {affinity.shape[0]}"
    assert affinity.shape[1] == cache_keys.shape[0], f"Affinity column count mismatch: expected {cache_keys.shape[0]}, got {affinity.shape[1]}"
    
    # Convert both tensors to the same dtype (float32) before matrix multiplication
    exp_term = ((-1) * (best_beta - best_beta * affinity)).exp().float()
    cache_values_float = cache_values.float()
    
    # Shape assertion for cache_logits calculation
    assert exp_term.shape[1] == cache_values_float.shape[0], f"Shape mismatch: exp_term {exp_term.shape}, cache_values {cache_values_float.shape}"
    
    cache_logits = exp_term @ cache_values_float
    
    tip_logits = clip_logits + cache_logits * best_alpha
    test_labels = test_labels.to(device)
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Log shapes for debugging
    print(f"Debug - run_tip_adapter_F shapes: cache_keys: {cache_keys.shape}, cache_values: {cache_values.shape}")
    print(f"Debug - run_tip_adapter_F shapes: val_features: {val_features.shape}, test_features: {test_features.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enforce float32 dtype
    cache_keys = cache_keys.to(device).float()
    cache_values = cache_values.to(device).float()
    val_features = val_features.to(device).float()
    test_features = test_features.to(device).float()
    
    # Enable the cached keys to be learnable
    # Enable the cached keys to be learnable
    adapter = nn.Linear(in_features=cache_keys.shape[1], out_features=cache_keys.shape[0], bias=False).to(clip_model.dtype).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Set weight data with correct shape: cache_keys is [100, 1024], which is already in the correct shape for nn.Linear weight
    adapter.weight.data = cache_keys.clone()
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        adapter = adapter.float().to(device)
        clip_weights = clip_weights.float().to(device)
        cache_values = cache_values.float().to(device)

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(device).float()

                
            affinity = adapter(image_features)
            
            # Shape assertion after adapter
            assert affinity.shape[0] == image_features.shape[0], f"Affinity row count mismatch: expected {image_features.shape[0]}, got {affinity.shape[0]}"
            assert affinity.shape[1] == cache_values.shape[0], f"Affinity column count mismatch: expected {cache_values.shape[0]}, got {affinity.shape[1]}"
            
            # Convert both tensors to the same dtype (float32) before matrix multiplication
            exp_term = ((-1) * (beta - beta * affinity)).exp().float()
            cache_values_float = cache_values.float()
            
            # Shape assertion for cache_logits calculation
            assert exp_term.shape[1] == cache_values_float.shape[0], f"Shape mismatch: exp_term {exp_term.shape}, cache_values {cache_values_float.shape}"
            
            cache_logits = exp_term @ cache_values_float
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()
        test_features = test_features.float().to(device)
        affinity = adapter(test_features)
        
        # Shape assertion after adapter
        assert affinity.shape[0] == test_features.shape[0], f"Affinity row count mismatch: expected {test_features.shape[0]}, got {affinity.shape[0]}"
        assert affinity.shape[1] == cache_values.shape[0], f"Affinity column count mismatch: expected {cache_values.shape[0]}, got {affinity.shape[1]}"
        
        # Convert both tensors to the same dtype (float32) before matrix multiplication
        exp_term = ((-1) * (beta - beta * affinity)).exp().float()
        cache_values_float = cache_values.float()
        
        # Shape assertion for cache_logits calculation
        assert exp_term.shape[1] == cache_values_float.shape[0], f"Shape mismatch: exp_term {exp_term.shape}, cache_values {cache_values_float.shape}"
        
        cache_logits = exp_term @ cache_values_float
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        test_labels = test_labels.to(device)
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")
   
    cache_logits = exp_term @ cache_values_float
    
    tip_logits = clip_logits + cache_logits * best_alpha
    test_labels = test_labels.to(device)
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(cfg['backbone'], device=device)
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)


    # === Custom cache build with TIMO ===
    print("\nConstructing CUSTOM cache model with TIMO...")
    C = len(dataset.classnames); shot = cfg['shots']
    
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_loader_cache, desc='Encoding support'):
            imgs = images.to(device)
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
            all_labels.append(labels.to(device))
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # reshape to [C, shot, D]
    support_feats = all_feats.view(C, shot, -1)

    # 2) Raw prompt embeddings [C, P, D]
    prompt_texts = []
    for name in dataset.classnames:
        text = name.replace('_', ' ')
        for t in dataset.template:
            prompt_texts.append(t.format(text))
    prompt_tokens = clip.tokenize(prompt_texts, truncate=True).to(device)
    with torch.no_grad():
        text_feats_all = clip_model.encode_text(prompt_tokens)
        text_feats_all = text_feats_all / text_feats_all.norm(dim=-1, keepdim=True)
    P = len(dataset.template)
    raw_text_feats = text_feats_all.view(C, P, -1)

    # 3) IGT refinement -> [C, D]
    F_IGT = refine_prompts(support_feats, raw_text_feats)

    # 4) TGI + build cache keys/values
    keys_list, vals_list = [], []
    for c in range(C):
        img_feats = support_feats[c]                 # [shot, D]
        adapted = refine_images(img_feats, F_IGT[c])  # [shot, D]
        proto = adapted.mean(dim=0, keepdim=True)    # [1, D]
        keys_list.append(proto)
        vals_list.append(F.one_hot(torch.tensor([c]), C).float())
    cache_keys = torch.cat(keys_list, dim=0)        # [C, D]
    cache_values = torch.cat(vals_list, dim=0)      # [C, C]
    print(f"Custom cache built: keys {cache_keys.shape}, values {cache_values.shape}")

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F)
           

if __name__ == '__main__':
    main(
    )