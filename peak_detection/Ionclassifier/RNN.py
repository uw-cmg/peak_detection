import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class IonRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=3, num_classes=118, dropout = 0.3):
        super(IonRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # RNN layers
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, # yes
            dropout = 0.3  # Add dropout for regularization

        )
        # Additional layers for better feature extraction
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8)
        self.norm = nn.LayerNorm(hidden_size * 2)

        # Multiple fully connected layers with batch normalization
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, lengths):

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Unpack the sequence
        output, _ = pad_packed_sequence(out, batch_first=True)

        # Apply attention
        out = out.permute(1, 0, 2)  # Change shape for attention
        attn_out, _ = self.attention(out, out, out)
        out = attn_out.permute(1, 0, 2)  # Change shape back
        out = self.norm(out)

        # Fully connected layers
        out = self.fc1(out)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Custom weighted loss function
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

### Just reference
# Training function with mixed precision training
def train_model(model, train_loader, criterion, optimizer, device, num_epochs, scaler):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_spectra, batch_labels in train_loader:
            batch_spectra = batch_spectra.to(device)
            batch_labels = batch_labels.to(device)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(batch_spectra, batch_lengths)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), batch_labels.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 2)
            total += batch_labels.size(0) * batch_labels.size(1)
            correct += (predicted == batch_labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')


def main():
    # Hyperparameters
    input_size = 1
    hidden_size = 256  # Increased for more complex patterns
    num_layers = 3
    num_classes = 118  # Number of chemical elements
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Your data loading here
    # Example:
    # spectra = load_spectra_data()
    # labels = load_label_data()

    # Create dataset and dataloader
    dataset = MassSpectraDataset(spectra, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MassSpectraRNN(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = WeightedFocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs, scaler)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': dataset.label_encoder,
    }, 'mass_spectra_rnn.pth')


# Prediction function with element names
def predict_elements(model, spectrum, label_encoder, device):
    model.eval()
    with torch.no_grad():
        spectrum = torch.FloatTensor(spectrum).unsqueeze(0).to(device)
        outputs = model(spectrum)
        probabilities = F.softmax(outputs, dim=2)
        predictions = torch.argmax(outputs, dim=2)

        # Convert numerical predictions to element names
        element_predictions = label_encoder.inverse_transform(predictions.cpu().numpy().ravel())
        confidence_scores = torch.max(probabilities, dim=2)[0].cpu().numpy().ravel()

        return element_predictions, confidence_scores