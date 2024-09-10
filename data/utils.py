from .dataset import UnifiedSFTDataset
from .collator import SFTDataCollator


def prepare_dataset(file, tokenizer, max_seq_length, template):
    dataset = UnifiedSFTDataset(file=file, tokenizer=tokenizer, max_seq_length=max_seq_length, template=template)
    collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=max_seq_length)
    return dataset, collator

