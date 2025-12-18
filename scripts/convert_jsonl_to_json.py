import json
import argparse

def convert_jsonl(input_file, chat_output, alpaca_output):
    chat_conversations = []
    alpaca_records = []

    role_map = {
        "system": "system",
        "user": "human",
        "assistant": "gpt"
    }

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = data.get("messages", [])
            converted = []

            for m in messages:
                role = role_map.get(m.get("role", ""), "")
                content = m.get("content", "")
                if role in {"human", "gpt"} and content:
                    converted.append({"from": role, "value": content})

            # Must be strict human → gpt alternation
            if len(converted) < 2 or len(converted) % 2 != 0:
                continue

            valid = True
            for i, msg in enumerate(converted):
                expected = "human" if i % 2 == 0 else "gpt"
                if msg["from"] != expected:
                    valid = False
                    break

            if not valid:
                continue

            # 1️⃣ Chat dataset (for Llama)
            chat_conversations.append({"conversations": converted})

            # 2️⃣ Alpaca dataset (for Mistral)
            for i in range(0, len(converted), 2):
                alpaca_records.append({
                    "instruction": converted[i]["value"],
                    "input": "",
                    "output": converted[i + 1]["value"]
                })

    # Write outputs
    with open(chat_output, "w", encoding="utf-8") as f:
        json.dump(chat_conversations, f, indent=2, ensure_ascii=False)

    with open(alpaca_output, "w", encoding="utf-8") as f:
        json.dump(alpaca_records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--chat_out", default="data_chat.json")
    parser.add_argument("--alpaca_out", default="data_alpaca.json")
    args = parser.parse_args()

    convert_jsonl(args.input_file, args.chat_out, args.alpaca_out)
