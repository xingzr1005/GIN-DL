import torch
import torch.nn as nn
import torch.nn.functional as F


class SGravity(nn.Module):
    def __init__(self, input_dim, hidden_dims, initial_distance_weight=0.6, dropout_prob=0.1):
        super(SGravity, self).__init__()
        self.distance_weight = nn.Parameter(
            torch.tensor(float(initial_distance_weight), dtype=torch.float32)
        )

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, origin_features, dest_features, distances, total_od):
        num_nodes, feature_dim = origin_features.shape
        origin_features = origin_features.float()
        dest_features = dest_features.float()
        distances = distances.float()
        total_od = total_od.float()

        origin_features = origin_features.unsqueeze(1).expand(num_nodes, num_nodes, feature_dim)
        dest_features = dest_features.unsqueeze(0).expand(num_nodes, num_nodes, feature_dim)
        distance_term = torch.exp(-self.distance_weight * distances).unsqueeze(-1)

        combined_features = torch.cat(
            [total_od, origin_features, dest_features, distance_term],
            dim=-1,
        )
        combined_features = combined_features.reshape(-1, combined_features.shape[-1])

        scores = self.network(combined_features).reshape(num_nodes, num_nodes)
        return F.softmax(scores, dim=1)


class GINDL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        num_nodes: int,
        poi_dim: int,
    ):
        super(GINDL, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_dim, out_features=1, bias=True),
            nn.LeakyReLU(),
        )

        gravity_input_dim = 1 + 2 * poi_dim + 1
        self.gravity_model = SGravity(input_dim=gravity_input_dim, hidden_dims=[256, 128])

    def forward(self, x_seq: torch.Tensor, poi_features, distance_matrix, total_od):
        batch_size, seq_len, num_nodes, _, channels = x_seq.shape
        x_seq = x_seq.sum(dim=3)
        lstm_in = x_seq.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_nodes,
            seq_len,
            channels,
        )

        lstm_out, _ = self.lstm(lstm_in)
        lstm_out_last = lstm_out[:, -1, :].reshape(
            batch_size,
            self.num_nodes,
            self.lstm_hidden_dim,
        )
        fc_out = self.fc_out(lstm_out_last)

        probabilities = self.gravity_model(
            poi_features,
            poi_features,
            distance_matrix,
            total_od,
        )

        result = (fc_out * probabilities).unsqueeze(-1).unsqueeze(1)
        return result, probabilities
