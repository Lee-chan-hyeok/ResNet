import os
import matplotlib.pyplot as plt

def save_loss_accuracy_graph(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, save_path):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, valid_loss_list, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list, label="Train Accuracy")
    plt.plot(epochs, valid_accuracy_list, label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    
    save_path = os.path.join(save_path, "loss_accuracy_graph.png")
    plt.savefig(save_path)
    plt.close()


def save_lr_graph(lr_list: list, save_path):
    epochs = range(1, len(lr_list) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, lr_list, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_path, exist_ok=True)
    
    save_path = os.path.join(save_path, "lr_graph.png")
    plt.savefig(save_path)
    plt.close()