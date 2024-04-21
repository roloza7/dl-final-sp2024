from torch.utils.data import Dataset
import torch
import torchvision.io as io
from typing import Callable
from collections import defaultdict
import random
import os
import json
from utils.transforms import get_transform
import warnings
from transformers import BertTokenizerFast
from transformers import PreTrainedTokenizerFast
import torch.nn as nn
from functools import partial
import pickle

def __collate_fn(data : list[tuple[torch.Tensor, torch.Tensor]], pad_id : int) -> tuple[torch.Tensor, torch.Tensor]:
    images, captions = list(zip(*data))
    captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=pad_id)
    images = torch.stack(images, dim=0)
    return images, captions

class COCOAEDataset(Dataset):
    def __init__(self,
                 root : str = None, 
                 annFile : str = None,
                 transform : Callable[[torch.Tensor, str|torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
                 tokenizer : PreTrainedTokenizerFast = None,
                 ignore_cache = False,
                 cache_dir = 'cache/',
                 train : bool = True) -> None:
        
        super().__init__()
        if root == None or annFile == None:
            raise ValueError("Root and annFile must be present for COCOAEDataset initialization")
        
        if tokenizer == None:
            warnings.warn("It's not advisable for the tokenizer to be None, since transforms expect the string to be tokenized")

        self.tokenizer = tokenizer
        
        self.root = root
        self.annfile = annFile
        self.transform = transform
        self.train = train
        self.ignore_cache = ignore_cache
        self.cache_dir = cache_dir

        with open(annFile, "r") as fp:
            self.dataset = json.load(fp)
        
        self.images_to_annotations : dict[int, list] = defaultdict(list)
        self.images : dict[int, tuple[int, str]] = {}
        self.__cache_or_load_vocab()

    def __cache_or_load_vocab(self):
        if os.path.exists(self.cache_dir) == False:
            os.mkdir(self.cache_dir)

        cache_path = os.path.join(self.cache_dir, 'tokenized_annotations.pkl')
        
        if os.path.exists(cache_path) and self.ignore_cache == False:
            print("Loading cached annotations...")
            with open(cache_path, "rb") as fp:
                cached = pickle.load(fp)
                self.images_to_annotations = cached['itoa']
                self.images = cached['img']
            del self.dataset
            return

        with open(cache_path, "wb") as fp:
            print("Creating annotations, might take a while...")
            self.__create_index()
            cached = {'itoa': self.images_to_annotations, 'img': self.images}
            print("Caching annotations...")
            pickle.dump(cached, fp)

    def __create_index(self):
        for ann in self.dataset['annotations']:
            tokenized_caption = self.tokenizer.encode(ann['caption'],
                                                      return_tensors='pt',
                                                      add_special_tokens=True).squeeze() if self.tokenizer is not None else ann['caption']
            self.images_to_annotations[ann['image_id']].append(tokenized_caption)

        for img in self.dataset['images']:
            self.images[len(self.images)] = (img['id'], img['file_name'])

        del self.dataset
        
    def __getitem__(self, index):

        image_id, image_path = self.images[index]
        image_path = os.path.join(self.root, image_path) 

        # Load image
        image = io.read_image(image_path)

        # Pick caption
        caption = random.choice(self.images_to_annotations[image_id])

        if self.transform != None:
            image, caption = self.transform(image, caption)

        return image, caption
    
    def __len__(self):
        return len(self.images)


# Example of how to use this dataset
if __name__ == "__main__":

    tokenizer : BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir='cache/')
    train_dataset = COCOAEDataset(root="coco/images/train2017", annFile="coco/annotations/ann2017/captions_train2017.json", transform=get_transform(), tokenizer=tokenizer, ignore_cache=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=partial(__collate_fn, pad_id=tokenizer.pad_token_id))
