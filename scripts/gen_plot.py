import matplotlib.pyplot as plt
import os

train_loss = [1.1044, 1.1042, 1.1029, 1.1003, 1.1053, 1.1009, 1.0999, 1.1061, 1.0985, 1.0952, 1.0943, 1.0824, 1.0736, 1.0412, 1.0403, 1.0345, 1.0297, 1.0565]
val_loss = [1.0930, 1.0926, 1.0923, 1.0921, 1.0919, 1.0926, 1.0921, 1.0917, 1.0916, 1.0913, 1.0939, 1.0941, 1.0950, 1.1112, 1.1441, 1.1209, 1.2100, 1.1299]
train_auroc = [0.4278, 0.4456, 0.4895, 0.5055, 0.4513, 0.5009, 0.5076, 0.4148, 0.5415, 0.5771, 0.5543, 0.6239, 0.6062, 0.6579, 0.6469, 0.6798, 0.6806, 0.6314]
val_auroc = [0.5616, 0.6011, 0.6237, 0.6373, 0.6045, 0.5537, 0.5898, 0.5989, 0.5706, 0.5514, 0.4746, 0.4723, 0.4689, 0.4870, 0.5062, 0.5424, 0.4181, 0.4915]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_auroc, label='Train')
plt.plot(val_auroc, label='Validation')
plt.title('AUROC History')
plt.legend()

plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/training_curves.png")
plt.close()
