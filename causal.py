import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_normalize(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class CausalInterventionModule(nn.Module):
    def __init__(self, feature_dim, embedding_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        self.Wq = nn.Linear(feature_dim, embedding_dim)
        self.Wk = nn.Linear(feature_dim, embedding_dim)

        self.causal_score = nn.Linear(2 * feature_dim, 1)
        self.ln = nn.LayerNorm(2 * feature_dim, eps=1e-4)  # 或 1e-3

        nn.init.normal_(self.Wq.weight, std=0.01);
        nn.init.constant_(self.Wq.bias, 0)
        nn.init.normal_(self.Wk.weight, std=0.01);
        nn.init.constant_(self.Wk.bias, 0)
        nn.init.normal_(self.causal_score.weight, std=0.001);
        nn.init.constant_(self.causal_score.bias, 0)

    def forward(self, cause_feat, query_feat, confounder_dict, prior_prob, tau=0.3):

        cause_feat = safe_normalize(cause_feat, dim=-1)
        query_feat = safe_normalize(query_feat, dim=-1)
        confounder_dict = safe_normalize(confounder_dict, dim=-1)

        q = self.Wq(query_feat)
        k = self.Wk(confounder_dict)

        q = safe_normalize(q, dim=-1)
        k = safe_normalize(k, dim=-1)

        attn_scores = torch.mm(q, k.t()) / max(tau, 1e-6)
        attention = F.softmax(attn_scores, dim=1)
        prior = prior_prob.clamp(min=0)
        prior = prior / (prior.sum() + 1e-12)  # (K,)

        z = torch.matmul(attention, confounder_dict)
        xz = torch.cat((cause_feat, z), dim=1)
        xz_norm = self.ln(xz)

        xz_activated = torch.tanh(xz_norm)

        causal_logit = self.causal_score(xz_activated).squeeze(-1)

        return causal_logit
