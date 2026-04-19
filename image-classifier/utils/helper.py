#helper.py

import torch
import matplotlib.pyplot as plt

def perform_eval(net, test_gen, loss_func, compute_dev):
    net.eval()
    running_loss = 0.0
    correct_cnt = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_gen:
            inputs, targets = inputs.to(compute_dev), targets.to(compute_dev)
            preds = net(inputs)
            error = loss_func(preds, targets)
            running_loss += error.item()
            _, top_pred = torch.max(preds.data, 1)
            total_samples += targets.size(0)
            correct_cnt += (top_pred == targets).sum().item()

    final_loss = running_loss / len(test_gen)
    final_acc = 100 * correct_cnt / total_samples
    return final_loss, final_acc

def visualize_metrics(log_data, output_img=None):
    steps = range(1, len(log_data['loss_history']) + 1)
    plt.figure(figsize=(10, 4))

    # Error tracking plot
    plt.subplot(1, 2, 1)
    plt.plot(steps, log_data['loss_history'], 'r-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Accuracy tracking plot
    plt.subplot(1, 2, 2)
    plt.plot(steps, log_data['acc_history'], 'g-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    
    if output_img:
        plt.savefig(output_img)
    
    plt.show()
    plt.close()
