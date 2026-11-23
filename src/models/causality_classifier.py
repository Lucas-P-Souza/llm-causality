from torch import nn
from transformers import AutoModel

PROJECTION_SIZE = 1024


class CausalityClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1, num_features=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.llm_projection = nn.Linear(self.hidden_size, PROJECTION_SIZE)
        self.feature_projection = (
            nn.Linear(num_features, PROJECTION_SIZE) if num_features else None
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(PROJECTION_SIZE, 512),
            nn.GELU(),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask, features=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # average in fp32 even when the encoder runs under bf16 autocast
        hidden = hidden.float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        fused = self.llm_projection(pooled)
        if self.feature_projection is not None:
            fused = fused + self.feature_projection(features)
        return self.classifier(fused)
