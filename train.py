import wandb
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import holocron
from trainer import ClassificationTrainer
from cnn_model import cnn_model
from iqa_dataset import IQA_DATASET


def build_dataset(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    size = config['image_size']
    train_transforms = transforms.Compose([
                  transforms.RandomCrop(224),
                  transforms.RandomRotation(degrees=2),
                  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize
              ])

    val_transforms = transforms.Compose([
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

    train_folder = config['train_folder']
    test_folder = config['test_folder']

    dsTrain = IQA_DATASET(train_folder, train_transforms)
    dsTest = IQA_DATASET(test_folder, val_transforms)
  
    train_loader = DataLoader(dsTrain, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dsTest, batch_size=config['batch_size'], shuffle=True)

    return train_loader, test_loader


def build_network(config):
    model_cut = -2
    num_classes=2
    lin_features=512
    dropout_prob=0.5
    bn_final=False
    concat_pool=True

    model_arch = config['model_arch']
    print(model_arch)
    base_model = holocron.models.__dict__[model_arch](False)

    if model_arch[:6]=='rexnet':
        nb_features = base_model.head[1].in_features

    elif model_arch[:6]=='resnet':
        nb_features = base_model.head.in_features

    else:
        nb_features=1024 #darknet



    model = cnn_model(base_model, model_cut, nb_features, num_classes,
                    lin_features, dropout_prob, bn_final=bn_final, concat_pool=concat_pool)

 
    #load checkpoint

    if model_arch =='rexnet1_0x':
        cp= 'checkpoints/rexnet1_0x.pth'

    elif model_arch =='rexnet1_3x':
        cp = 'checkpoints/rexnet1_3x.pth'

    elif model_arch == "darknet19":
        cp = 'checkpoints/darknet19.pth'


    model.load_state_dict(torch.load(cp,map_location=torch.device('cpu')))


    return model


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, test_loader = build_dataset(config)
        model = build_network(config)
        lr = config['lr']
        wd = config['wd']

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Create the contiguous parameters.
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = holocron.optim.RAdam(model_params, lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=wd)


        #Trainer
        trainer = ClassificationTrainer(model, train_loader, test_loader, criterion, optimizer, 0, output_file=config['checkpoint'], configwb=True)

        trainer.fit_n_epochs(config['epochs'], config['lr'], config['freeze'])
