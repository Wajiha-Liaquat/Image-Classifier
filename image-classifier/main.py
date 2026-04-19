#main.py

import torch
import sys
import os
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath("image-classifier"))

from dataset.custom_dataset import CIFAR10Dataset
from models.model import CIFAR10CNN
from train.train import run_training
from utils.helper import perform_eval, visualize_metrics

def start_pipeline():
    compute_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Strictly 3 trials as per PDF
    trials = [
        {"rounds": 5,  "opt": "SGD",  "lr": 0.01,  "bs": 32},
        {"rounds": 10, "opt": "Adam", "lr": 0.001, "bs": 64},
        {"rounds": 20, "opt": "Adam", "lr": 0.001, "bs": 64}
    ]

    img_fx = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    top_score = 0.0

    for idx, conf in enumerate(trials):
        print(f"----- EXECUTING TRIAL {idx + 1} OF {len(trials)} -----")
        
        train_pack = CIFAR10Dataset('/content/final_dataset/train', transform_node=img_fx)
        test_pack = CIFAR10Dataset('/content/final_dataset/test', transform_node=img_fx)
        
        train_it = DataLoader(train_pack, batch_size=conf['bs'], shuffle=True)
        test_it = DataLoader(test_pack, batch_size=conf['bs'], shuffle=False)
        
        cnn_net = CIFAR10CNN()
        final_net, metrics = run_training(cnn_net, train_it, conf['rounds'], conf['lr'], conf['opt'], compute_dev)
        
        crit = torch.nn.CrossEntropyLoss()
        _, final_acc = perform_eval(final_net, test_it, crit, compute_dev)
        
        print(f"\n>> EXPERIMENT {idx + 1} COMPLETE | TEST ACC: {final_acc:.2f}% <<")
        
        graph_name = f"image-classifier/plots/trial_{idx + 1}_summary.png"
        visualize_metrics(metrics, output_img=graph_name)
        
        if final_acc > top_score:
            top_score = final_acc
            torch.save(final_net.state_dict(), "image-classifier/results/best_model.pth")

    print(f"\n{'='*35}")
    print("Experiments Finished Successfully!")
    print(f"{'='*35}")

if __name__ == "__main__":
    start_pipeline()
