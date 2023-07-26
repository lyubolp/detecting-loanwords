from consts import MAX_LENGTH

# function to convert a sentence to a tensor
def sentence_to_tensor(content, target_size = MAX_LENGTH) -> torch.Tensor:
    # Add padding to the end of the sentence, so that the length is equal to target_size
    tensor = torch.tensor(content, dtype=torch.long, device=device).view(-1, 1)

    if tensor.size()[0] < target_size:
        padding = torch.zeros(target_size - tensor.size()[0], 1, dtype=torch.int32, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
        
    return tensor