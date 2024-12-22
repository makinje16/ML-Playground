from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import data as hData

train_dataset, val_dataset, test_dataset = hData.GetDataset()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

class CaliHousingRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CaliHousingRegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.output(x)

        return x

model = CaliHousingRegressionModel(10, 25, 1)
criterion = nn.MSELoss()
optim = Adam(model.parameters(), lr=0.003)

epochs=25
for epoch in range(epochs):
    avg_loss = 0
    num_batches = 0
    for features, labels in train_loader:
        labels = labels.unsqueeze(1)

        # Forward pass
        predictions = model(features)

        loss = criterion(predictions, labels)
        avg_loss += loss.item()
        num_batches += 1
        # Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(f"Training Epoch {epoch+1}/{epochs}, AverageLoss: {avg_loss/num_batches:.4f}")

    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for features, labels in val_loader:
            labels = labels.unsqueeze(1)
            predictions = model(features)
            loss = criterion(predictions, labels)

            val_loss += loss.item()
            val_batches += 1
    avg_val_loss = val_loss / val_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")
    model.train()  # Switch back to training mode

model.eval()
test_loss = 0.0
test_batches = 0
with torch.no_grad():
    for features, labels in test_loader:
        labels = labels.unsqueeze(1)
        predictions = model(features)
        loss = criterion(predictions, labels)
        test_loss += loss.item()
        test_batches += 1

avg_test_loss = test_loss / test_batches
print(f"Test Loss: {avg_test_loss:.4f}")
