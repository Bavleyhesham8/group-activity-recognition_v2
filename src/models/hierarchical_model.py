import torch
import torch.nn as nn
import torch.nn.functional as F

def device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"

class PersonLSTM(nn.Module):
    def __init__(self, feat_dim=512, hidden=512, proj_dim=256,
                 num_action_classes=10, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.proj_dim = proj_dim
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.action_head = nn.Linear(hidden, num_action_classes)
        self.project = nn.Linear(hidden, proj_dim) if proj_dim is not None else None
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        if self.project is not None:
            nn.init.xavier_uniform_(self.project.weight)
            nn.init.zeros_(self.project.bias)

    def forward(self, x):
        # x: (B*K, T, F)
        x = x.float()
        h, _ = self.lstm(x)            # (B*K, T, hidden)
        h = self.norm(h)
        a_logits = self.action_head(h) # (B*K, T, num_action_classes)
        if self.project is not None:
            p = self.project(h)        # (B*K, T, proj_dim)
        else:
            p = h
        return h, p, a_logits


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.q(x).squeeze(-1)      # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, T)
        out = (attn.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return out


class TwoTeamGroupLSTM(nn.Module):
    def __init__(self, proj_dim=256, fc_dim=512, group_hidden=256,
                 num_group_classes=8, num_layers=1, dropout=0.3,
                 use_temporal_attn=True):
        super().__init__()
        self.proj_dim = proj_dim
        two_team_dim = proj_dim * 2
        self.pool_norm = nn.LayerNorm(two_team_dim)
        self.fc = nn.Sequential(
            nn.Linear(two_team_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(fc_dim, group_hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.use_temporal_attn = use_temporal_attn
        if use_temporal_attn:
            self.temporal_pool = TemporalAttentionPool(group_hidden, hidden=128)
            self.head = nn.Linear(group_hidden, num_group_classes)
        else:
            self.head = nn.Linear(group_hidden, num_group_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, person_proj, team_flags):
        # person_proj: (B, T, K, proj_dim)
        with torch.amp.autocast(device_type(), enabled=False):
            person_proj = person_proj.float()
            flags = team_flags.float().unsqueeze(1).unsqueeze(-1)  # (B,1,K,1)
            NEG = -1e6
            # Team A (flag=1)
            teamA = person_proj * flags + (1.0 - flags) * NEG
            teamA = teamA.max(dim=2)[0]  # (B, T, P)
            # Team B (flag=0)
            teamB = person_proj * (1.0 - flags) + flags * NEG
            teamB = teamB.max(dim=2)[0]  # (B, T, P)

            teamA = torch.nan_to_num(teamA, nan=0.0, neginf=0.0)
            teamB = torch.nan_to_num(teamB, nan=0.0, neginf=0.0)

            Z = torch.cat([teamA, teamB], dim=-1)  # (B, T, 2P)
            Z = self.pool_norm(Z)
            Z = self.fc(Z)                          # (B, T, fc_dim)
            g_seq, _ = self.lstm(Z)                 # (B, T, group_hidden)
            if self.use_temporal_attn:
                g_vec = self.temporal_pool(g_seq)   # (B, group_hidden)
            else:
                g_vec = g_seq[:, -1, :]
            logits = self.head(g_vec)
            return logits, g_seq


class HierarchicalModel(nn.Module):
    def __init__(self, feat_dim=512, person_hidden=512, proj_dim=256,
                 fc_dim=512, group_hidden=256,
                 num_action_classes=10, num_group_classes=8,
                 person_layers=2, group_layers=1, use_temporal_attn=True):
        super().__init__()
        self.person = PersonLSTM(feat_dim, person_hidden, proj_dim,
                                 num_action_classes, num_layers=person_layers)
        self.group = TwoTeamGroupLSTM(proj_dim, fc_dim, group_hidden,
                                      num_group_classes, num_layers=group_layers,
                                      use_temporal_attn=use_temporal_attn)

    def forward(self, features, team_flags, stage=2):
        # features: (B, T, K, F)
        B, T, K, F = features.shape
        x = features.permute(0, 2, 1, 3).reshape(B * K, T, F)
        with torch.amp.autocast(device_type(), enabled=False):
            h_raw, p_proj, a_logits = self.person(x)
        p_proj = p_proj.reshape(B, K, T, -1).permute(0, 2, 1, 3)  # (B, T, K, P)
        a_logits = a_logits.reshape(B, K, T, -1).permute(0, 2, 1, 3)  # (B, T, K, C)
        if stage == 1:
            return a_logits
        g_logits, g_seq = self.group(p_proj, team_flags)
        return g_logits, a_logits
