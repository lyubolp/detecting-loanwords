from dataset_utils import yield_tokens, sentence_to_tensor, load_files, build_vocab_transformation
from transcription_dataset import TranscriptionDataset

from guppy import hpy

files = load_files('/mnt/d/Projects/masters-thesis/data/transcriptions')

train_dataset = TranscriptionDataset(files, start_index=0)
# After loading files list
print('---After loading the dataset---')
h = hpy()
print({h.heap()})

