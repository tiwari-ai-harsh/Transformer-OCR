from imports_pytorch import *

class WordDataset(Dataset):
    def __init__(self,  transform=None):
        with open("../dataset/v011_labels_small.json") as f:
            data = json.load(f)
        self.transform = transform
        self.vocab = set()
        list_data = []
        for i, j in data.items():
            list_data.append(["../dataset/v011_words_small/"+i, j])
        list_data = np.array(list_data)
        text_data = list_data[:, 1]
        self.vocab.update(["<start>"])
        self.vocab.update(["<end>"])
        self.vocab.update(set("".join(text_data)))
        self.vocab_size = len(self.vocab) + 1

        self.char2idx = {u:i+1 for i, u in enumerate(self.vocab)}
        self.idx2char = {i+1: u  for i, u in enumerate(self.vocab)}

        text_as_int = [self._word_to_tensor(word) for word in text_data]
        # [[self.char2idx["<start>"]]+[self.char2idx[c] for c in text]+[self.char2idx["<end>"]] for text in text_data]

        self.padded_text      = self._pad_sequence(text_as_int)

        self.images           = list_data[:, 0]

    def __len__(self):
        return len(self.padded_text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        img = Image.open(self.images[idx])
        label = self.padded_text[idx]

        if self.transform:
            img = self.transform(img)
        # print(self._tensor_to_word(label))
        return img, torch.tensor(label)

    def _tensor_to_word(self, tensor):
        words=""
        for ten in tensor:
            if self.idx2char[ten] == "<start>":
                # words += self.idx2char[ten]
                pass
            elif self.idx2char[ten] == "<end>":
                break
            else:
                words += self.idx2char[ten]
        return words

    def _word_to_tensor(self, string):
        tensor = []
        tensor.append(self.char2idx["<start>"])
        for st in string:
            tensor.append(self.char2idx[st])
        tensor.append(self.char2idx["<end>"])
        return tensor

    def _pad_sequence(self, sequence):
        max_len = max(len(i) for i in sequence)

        for i in range(len(sequence)):
            sequence[i].extend((max_len - len(sequence[i]))*[0])
        return sequence

if __name__ == "__main__":
    wd = WordDataset(transform=transforms.Compose([transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor()]))
    plt.imshow(wd[1][0].permute(2,1,0))
    # plt.show()
    # print(wd[1][0].shape)