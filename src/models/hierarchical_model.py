import torch
import torch.nn as nn
import torch.nn.functional as F

class PersonLSTM(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=128, hidden_dim=256, num_frames=9):
        super(PersonLSTM, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (B * num_players, T, D)
        embedded = F.relu(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        
        # Temporal attention
        attn_weights = self.temporal_attention(lstm_out)
        attended_out = torch.sum(lstm_out * attn_weights, dim=1)
        
        return attended_out

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, x):
        # x shape: (B, num_players, hidden_dim)
        query = self.query.expand(x.size(0), -1, -1)
        attn_output, _ = self.attn(query, x, x)
        return attn_output.squeeze(1)

class TwoTeamGroupLSTM(nn.Module):
    def __init__(self, person_hidden_dim=256, group_hidden_dim=512):
        super(TwoTeamGroupLSTM, self).__init__()
        # Each team has 6 players
        self.team1_lstm = nn.LSTM(person_hidden_dim, group_hidden_dim, batch_first=True)
        self.team2_lstm = nn.LSTM(person_hidden_dim, group_hidden_dim, batch_first=True)
        
        self.team1_pool = TemporalAttentionPool(group_hidden_dim)
        self.team2_pool = TemporalAttentionPool(group_hidden_dim)

    def forward(self, person_features):
        # person_features shape: (B, 12, hidden_dim)
        team1 = person_features[:, :6, :]
        team2 = person_features[:, 6:, :]
        
        t1_out, _ = self.team1_lstm(team1)
        t2_out, _ = self.team2_lstm(team2)
        
        t1_pooled = self.team1_pool(t1_out)
        t2_pooled = self.team2_pool(t2_out)
        
        return torch.cat([t1_pooled, t2_pooled], dim=1)

class HierarchicalModel(nn.Module):
    def __init__(self, config):
        super(HierarchicalModel, self).__init__()
        self.num_players = config['num_players']
        self.num_frames = config['num_frames']
        
        self.person_lstm = PersonLSTM(
            input_dim=config['resnet_feature_dim'],
            embed_dim=config['person_embed_dim'],
            hidden_dim=config['person_hidden_dim'],
            num_frames=config['num_frames']
        )
        
        self.group_level = TwoTeamGroupLSTM(
            person_hidden_dim=config['person_hidden_dim'],
            group_hidden_dim=config['group_hidden_dim']
        )
        
        # Classifier for Stage 1 (Group Activity)
        self.group_classifier = nn.Linear(config['group_hidden_dim'] * 2, config['num_group_classes'])
        
        # Classifier for individual actions (if needed in Stage 2)
        self.person_classifier = nn.Linear(config['person_hidden_dim'], config['num_person_classes'])

    def forward(self, x):
        # x shape: (B, 12, 9, 2048)
        batch_size = x.size(0)
        
        # Flatten for PersonLSTM
        # (B * 12, 9, 2048)
        x_flat = x.view(batch_size * self.num_players, self.num_frames, -1)
        person_features = self.person_lstm(x_flat)
        
        # Reshape back: (B, 12, person_hidden_dim)
        person_features = person_features.view(batch_size, self.num_players, -1)
        
        # Group Level logic
        group_features = self.group_level(person_features)
        
        # Classify group activity
        group_logits = self.group_classifier(group_features)
        
        # Classify individual actions
        person_logits = self.person_classifier(person_features)
        
        return group_logits, person_logits
