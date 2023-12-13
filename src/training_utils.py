import torch
import src.constants as const

from src.transformer_model import create_mask



# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([const.BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([const.EOS_IDX])))


def train_epoch(model, optimizer, train_dataloader, loss_fn):
    model.train()
    losses = 0

    i = 0
    for src, tgt in train_dataloader:
        i += 1
        src = src.to(const.device)
        tgt = tgt.to(const.device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model, validation_dataloader, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in validation_dataloader:
        src = src.to(const.device)
        tgt = tgt.to(const.device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(validation_dataloader))

