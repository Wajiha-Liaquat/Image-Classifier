#train.py

import torch
import torch.nn as nn
import torch.optim as optim

def run_training(model_net, data_iter, round_count, step_rate, opt_type, device_id):
    # Setup loss and optimizer
    objective = nn.CrossEntropyLoss()
    
    if opt_type.lower() == 'adam':
        optimizer = optim.Adam(model_net.parameters(), lr=step_rate)
    else:
        optimizer = optim.SGD(model_net.parameters(), lr=step_rate, momentum=0.9)

    stats = {'loss_history': [], 'acc_history': []}
    model_net.to(device_id)

    for epoch_idx in range(round_count):
        model_net.train()
        total_loss, correct_hits, samples = 0.0, 0, 0

        for batch_data, labels in data_iter:
            batch_data, labels = batch_data.to(device_id), labels.to(device_id)

            # Standard optimization step
            optimizer.zero_grad()
            preds = model_net(batch_data)
            loss_val = objective(preds, labels)
            loss_val.backward()
            optimizer.step()

            # Metric tracking
            total_loss += loss_val.item()
            _, predicted = torch.max(preds.data, 1)
            samples += labels.size(0)
            correct_hits += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_iter)
        avg_acc = 100 * correct_hits / samples
        
        stats['loss_history'].append(avg_loss)
        stats['acc_history'].append(avg_acc)

        print(f"Epoch {epoch_idx+1}/{round_count} -> Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%")

    return model_net, stats
