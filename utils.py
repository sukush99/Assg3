from tqdm import tqdm
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    target = target.to(output.device)
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).float()  # Ensure float32
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).float()  # Ensure float32

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        # Ensure loaded tensors are float32
        cache_keys = cache_keys.float()
        cache_values = cache_values.float()

    # Log cache shapes for debugging
    print(f"Debug - Cache model shapes: keys: {cache_keys.shape}, values: {cache_values.shape}")
    
    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # Ensure features are float32
                image_features = image_features.float()
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
        # Ensure loaded features are float32
        features = features.float()
    
    # Log feature shape for debugging
    print(f"Debug - {split} features shape: {features.shape}, labels: {labels.shape}")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
        # Convert inputs to float32 at the start
        cache_keys = cache_keys.float()
        cache_values = cache_values.float()
        features = features.float()
        
        # Log shapes for debugging
        print(f"Debug - search_hp shapes: features: {features.shape}, cache_keys: {cache_keys.shape}, cache_values: {cache_values.shape}")
        
        # Shape assertions
        feature_dim = features.shape[1]
        assert cache_keys.shape[1] == feature_dim, f"Feature dimension mismatch: features {features.shape}, cache_keys {cache_keys.shape}"
        assert cache_values.shape[0] == cache_keys.shape[0], f"Class count mismatch: cache_keys {cache_keys.shape}, cache_values {cache_values.shape}"
        
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    # Fix: transpose cache_keys to match dimension requirements
                    affinity = features @ cache_keys.t()
                
                # Ensure all operations are in float32
                affinity = affinity.float()
                exp_term = ((-1) * (beta - beta * affinity)).exp()
                
                # Shape assertion for matrix multiplication
                assert exp_term.shape[1] == cache_values.shape[0], f"Shape mismatch: exp_term {exp_term.shape}, cache_values {cache_values.shape}"
                
                cache_logits = exp_term @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha
