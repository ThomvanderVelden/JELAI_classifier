import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


df = pd.read_csv(
    "questions_before_midterm.csv",
    sep=",",
    quotechar="'",
    escapechar="\\",
    engine="python",
)
tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

token_lengths = []
for text in df["question"]:
    tokens = tokenizer.tokenize(text)
    token_lengths.append(len(tokens))


plt.hist(token_lengths, bins=50)
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.title("Distribution of Token Lengths in Questions Dataset")
plt.show()

print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
print(f"Median token length: {sorted(token_lengths)[len(token_lengths)//2]}")
print(f"Maximum token length: {max(token_lengths)}")
print(
    f"Number of questions with more than 256 tokens: {sum(1 for l in token_lengths if l > 256)}"
)
