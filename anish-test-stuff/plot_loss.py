import matplotlib.pyplot as plt
import re
import os

def parse_file_multi_line(filename):
    """
    Parses the format where Epoch and Loss are on different lines.
    Used for: 3rdAttempt.txt
    Format: 
      Epoch 1 Completed.
      Train Loss: 0.0499 | Validation Loss: 0.0325
    """
    epochs = []
    train_losses = []
    val_losses = []
    
    current_epoch = None

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return [], [], []

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 1. Find the Epoch line
        epoch_match = re.search(r"Epoch (\d+) Completed", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        
        # 2. Find the Loss line (only if we have a current epoch)
        if current_epoch is not None:
            # Look for "Validation Loss" (spelled out)
            loss_match = re.search(r"Train Loss: ([\d\.]+) \| Validation Loss: ([\d\.]+)", line)
            if loss_match:
                epochs.append(current_epoch)
                train_losses.append(float(loss_match.group(1)))
                val_losses.append(float(loss_match.group(2)))
                current_epoch = None # Reset for next block

    return epochs, train_losses, val_losses

def parse_file_single_line(filename):
    """
    Parses the format where Epoch and Loss are on the SAME line.
    Used for: 5thAttempt.txt
    Format:
      Epoch 1 Done. Train Loss: 0.0404 | Val Loss: 0.0280
    """
    epochs = []
    train_losses = []
    val_losses = []

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return [], [], []

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Look for "Val Loss" (abbreviated)
        match = re.search(r"Epoch (\d+) Done. Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))

    return epochs, train_losses, val_losses

def plot_logs():
    # --- DATA LOADING ---
    file_old = '3rdAttempt.txt'
    file_new = '5thAttempt.txt'
    
    # Parse File 1 (Labeled as 2nd Attempt per instructions)
    epochs_2, train_2, val_2 = parse_file_multi_line(file_old)
    
    # Parse File 2 (5th Attempt)
    epochs_5, train_5, val_5 = parse_file_single_line(file_new)
    
    # Setup the plot
    plt.figure(figsize=(18, 6))

    # --- PLOT 1: 2nd Attempt (from 3rdAttempt.txt) ---
    plt.subplot(1, 3, 1)
    if epochs_2:
        plt.plot(epochs_2, train_2, label='Train Loss', marker='o', color='gray')
        plt.plot(epochs_2, val_2, label='Validation Loss', marker='x', color='orange', linestyle='--')
        plt.title('2nd Attempt (Old Dataset)')
        plt.xlabel('Epoch')
        plt.ylabel('L1 Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    else:
        plt.text(0.5, 0.5, f"{file_old} empty or missing", ha='center')

    # --- PLOT 2: 5th Attempt (New Dataset) ---
    plt.subplot(1, 3, 2)
    if epochs_5:
        plt.plot(epochs_5, train_5, label='Train Loss', marker='o', color='blue')
        plt.plot(epochs_5, val_5, label='Validation Loss', marker='x', color='red', linestyle='--')
        plt.title('5th Attempt (DF2K Dataset)')
        plt.xlabel('Epoch')
        plt.ylabel('L1 Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    else:
        plt.text(0.5, 0.5, f"{file_new} empty or missing", ha='center')

    # --- PLOT 3: Direct Comparison (Validation Loss) ---
    plt.subplot(1, 3, 3)
    if epochs_2 and epochs_5:
        plt.plot(epochs_2, val_2, label='2nd Attempt Val', linestyle='--', color='orange')
        plt.plot(epochs_5, val_5, label='5th Attempt Val', color='red', linewidth=2)
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('L1 Loss (Lower is Better)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_graphs.png')
    print("Graphs saved to comparison_graphs.png")
    
    # Try to show, handle error if on headless server
    try:
        plt.show()
    except UserWarning:
        pass

if __name__ == "__main__":
    plot_logs()