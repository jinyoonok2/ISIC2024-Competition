import torch.nn as nn
import torch
from torch.nn import Parameter

class SimpleHead(nn.Module):
    def __init__(self, num_features):
        super(SimpleHead, self).__init__()
        # The original SimpleHead with a single linear layer and sigmoid activation
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

class SimpleHeadV2(nn.Module):
    def __init__(self, num_features):
        super(SimpleHeadV2, self).__init__()

        # The updated SimpleHeadV2 with selective batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # Final layer reducing to 1 output
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.classifier(x)

class SpatialAveragePooling(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)

class h_swish(nn.Module):
    def forward(self, x):
        return x * nn.functional.relu6(x + 3, inplace=True) / 6


class SCSA(nn.Module):
    def __init__(self, channel=512, G=8, device=None):
        super(SCSA, self).__init__()
        self.G = G
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_s = SpatialAveragePooling()
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Initialize cweight and cbias on the correct device
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1, device=device))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1, device=device))

        self.conv1 = nn.Conv2d(channel // (2 * G), channel // (2 * G), kernel_size=1, stride=1, padding=0)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.act = h_swish()

        self.conv_h = nn.Conv2d(channel // (2 * G), channel // (2 * G), kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(channel // (2 * G), channel // (2 * G), kernel_size=1, stride=1, padding=0)
        self.conv_s = nn.Conv2d(channel // (2 * G), channel // (2 * G), kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x = x.view(n * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.gap(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_h = self.pool_h(x_1)
        x_w = self.pool_w(x_1).permute(0, 1, 3, 2)
        x_s = self.pool_s(x_1)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.gn(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)

        a_hws = a_h * a_w * x_s
        x_spatial = x_1 * a_hws.sigmoid()

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(n, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)

        return out


class SCSAHead(nn.Module):
    def __init__(self, input_channels, backbone_output_size, device):
        super(SCSAHead, self).__init__()
        # Initialize the SCSA module
        self.scsa = SCSA(input_channels).to(device)

        # Pass the backbone output size (after passing through the modified backbone) to initialize the fully connected layers
        with torch.no_grad():
            scsa_output = self.scsa(backbone_output_size)

        # Flatten scsa_output and get the dimension of the flattened tensor excluding batch size
        flattened_dim = torch.flatten(scsa_output, start_dim=1).size(1)

        # Initialize fully connected layers based on calculated output size
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_dim, 1024),  # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(1024, 512),  # Second fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
        ).to(device)

        self.output_layer = nn.Linear(512, 1).to(device)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.scsa(x)  # Apply SCSA module
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.fc_layers(x)  # Apply the fully connected layers
        x = self.output_layer(x)  # Output layer
        x = self.sigmoid(x)  # Sigmoid activation
        return x





def get_model_head(head_type, input_channels, backbone_output_size, device):
    if head_type.lower() == 'simplehead':
        return SimpleHead(input_channels).to(device)
    elif head_type.lower() == 'simpleheadv2':
        return SimpleHeadV2(input_channels).to(device)
    elif head_type.lower() == 'scsa':
        return SCSAHead(input_channels, backbone_output_size, device).to(device)
    else:
        raise ValueError(f"Unknown head type: {head_type}. Please choose 'simplehead', 'simpleheadv2', or 'scsa'.")



