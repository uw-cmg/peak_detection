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
        out0, _ = self.rnn(packed)
        # Unpack the sequence
        output, _ = pad_packed_sequence(out0, batch_first=True)

        # # Apply attention
        # out = out.permute(1, 0, 2)  # Change shape for attention
        # attn_out, _ = self.attention(out, out, out)
        # out = attn_out.permute(1, 0, 2)  # Change shape back
        # out = self.norm(out)

        # Create attention mask for padding
        max_len = output.size(1)
        attention_mask = torch.arange(max_len).expand(len(lengths), max_len) >= torch.tensor(lengths).unsqueeze(1)
        attention_mask = attention_mask.to(output.device)

        # Apply attention
        # Change shape from (batch, seq, features) to (seq, batch, features)
        output = output.transpose(0, 1)

        # Apply attention with mask
        attn_out, _ = self.attention(
            output, output, output,
            key_padding_mask=attention_mask
        )

        # Change shape back to (batch, seq, features)
        out = attn_out.transpose(0, 1)
        # # Add residual connection around attention
        # out = attn_out.transpose(0, 1) + output

        # Layer normalization
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
"""

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 2)
    total += batch_labels.size(0) * batch_labels.size(1)
    correct += (predicted == batch_labels).sum().item()

accuracy = 100 * correct / total
print(
    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
Consider using position encodings:

pythonCopyself.pos_encoder = PositionalEncoding(hidden_size * 2)
output = self.pos_encoder(output)


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

# Monitor attention weights if needed:
pythonCopyattn_out, attn_weights = self.attention(
    output, output, output,
    key_padding_mask=attention_mask,
    need_weights=True
)
"""