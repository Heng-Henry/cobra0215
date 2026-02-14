from datasets import load_dataset

# LongBench config names:
# - multihop (2wikimultihopqa) 對應 config "2wikimqa"
# - hotpotqa         對應 config "hotpotqa"

ds = load_dataset("THUDM/LongBench", "2wikimqa")["test"]  # multihop QA
hotpot = load_dataset("THUDM/LongBench", "hotpotqa")["test"]
print(ds[0]["input"][:200], ds[0]["answers"])  # input 是拼接長文，answers 是標準答案列表
