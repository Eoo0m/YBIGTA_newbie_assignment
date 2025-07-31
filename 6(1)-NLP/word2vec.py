import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"],
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        token_ids = tokenizer(
            corpus, padding=False, truncation=True, add_special_tokens=False
        )["input_ids"]
        token_ids = [
            item
            for sublist in token_ids
            for item in sublist
            if item != tokenizer.pad_token_id
        ]

        for epoch in range(num_epochs):
            print(
                f"Starting Word2Vec training with method: {self.method}, epochs: {num_epochs}"
            )

            if self.method == "cbow":
                self._train_cbow(token_ids, optimizer, criterion)
            else:
                self._train_skipgram(token_ids, optimizer, criterion)

    def _train_cbow(
        self,
        token_ids: list[int],
        optimizer: Adam,
        criterion,
    ) -> None:
        window = self.window_size
        for center_idx in range(window, len(token_ids) - window):
            if center_idx % 1000 == 0:
                print(f"Progress: {center_idx}/{len(token_ids)}")
            context = (
                token_ids[center_idx - window : center_idx]
                + token_ids[center_idx + 1 : center_idx + window + 1]
            )
            target = token_ids[center_idx]

            context_tensor = torch.tensor(context).long()
            target_tensor = torch.tensor([target]).long()

            context_embeds = self.embeddings(context_tensor)
            context_mean = context_embeds.mean(dim=0)

            logits = self.weight(context_mean)
            loss = criterion(logits.unsqueeze(0), target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _train_skipgram(self, token_ids: list[int], optimizer: Adam, criterion) -> None:
        window = self.window_size
        for center_idx in range(window, len(token_ids) - window):
            print(f"Progress: {center_idx}/{len(token_ids)}")
            center = token_ids[center_idx]
            context = (
                token_ids[center_idx - window : center_idx]
                + token_ids[center_idx + 1 : center_idx + window + 1]
            )

            for ctx in context:
                center_tensor = torch.tensor([center]).long()
                ctx_tensor = torch.tensor([ctx]).long()

                center_embed = self.embeddings(center_tensor)
                logits = self.weight(center_embed.squeeze(0))

                loss = criterion(logits.unsqueeze(0), ctx_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
