from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
import json
from torch.utils.data import Dataset
import torch
from tqdm.auto import tqdm
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer


class CodeDataset(Dataset):
    def __init__(self, lines) -> None:
        super().__init__()
        self.lines = lines
        self.tokenizer = WordPunctTokenizer()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index) -> T_co:
        data = json.loads(self.lines[index])
        code = data["code"]
        tokens = self.tokenizer.tokenize(code)
        if len(tokens) > 512:
            del self.lines[index]
            return self.__getitem__(index)
        return " ".join(tokens)


pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_small_code_comment_generation_java"),
    tokenizer=AutoTokenizer.from_pretrained(
        "SEBIS/code_trans_t5_small_code_comment_generation_java", use_fast=False, skip_special_tokens=True
    ),
    device="cuda:0",
)
with open("data/codet5/test.jsonl", "r") as f:
    lines = f.readlines()

dataset = CodeDataset(lines)
batch_size = 1
with open("results/codet5.txt", "w") as f:
    id = 0
    tokenizer = RegexpTokenizer(r"\w+")
    for out in tqdm(pipeline(dataset, batch_size=batch_size), total=len(dataset)):
        true_comment = json.loads(lines[id])["label"]
        output = tokenizer.tokenize(out[0]["summary_text"].lower())
        predicted_comment = "|".join(output)
        print(true_comment, predicted_comment, file=f)
        id += 1
        torch.cuda.empty_cache()
