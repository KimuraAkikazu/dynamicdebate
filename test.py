from datasets import load_dataset

# データセットをロード
ds = load_dataset("cais/mmlu", "all", split="test")

# 最初の10問を処理
for i, example in enumerate(ds):
    if i >= 10:  # 10問で終了
        break
    question = example["question"]
    options = example["choices"]
    correct_label = example["answer"]  # 'A', 'B', 'C', 'D'

    # 議論トピックとして問題文と選択肢を整形
    topic_text = f"{question}\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\n回答はA〜Dの中から1つ選んでください。"
    print(topic_text)
    print(f"正解: {correct_label}\n")
