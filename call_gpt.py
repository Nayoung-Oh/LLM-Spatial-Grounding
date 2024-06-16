from openai import OpenAI
from dotenv import load_dotenv
import json
# pip install python-dotenv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_relation", type=int)
parser.add_argument("--important_object_first", action="store_true")
parser.add_argument("--ours", action="store_true")
parser.add_argument("--logical", action="store_true")
parser.add_argument("--combined", action="store_true")
parser.add_argument("--vanilla", action="store_true")
parser.add_argument("--rewrite", action="store_true")

args = parser.parse_args()
important_object_first = args.important_object_first
for num_relation in range(3, 8):
    if args.important_object_first:
        folder_name = f"data/if_answers_{num_relation}"
    else:
        folder_name = f"data/answers_{num_relation}"
    if args.ours:
        folder_name += "_ours"
    if args.logical:
        folder_name += "_logical"
    if args.combined:
        folder_name += "_combined"
    if args.vanilla:
        folder_name += "_vanilla"
    if args.rewrite:
        folder_name += "_rewrite"
    os.makedirs(folder_name, exist_ok=True)
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def get_completion(prompt, model="gpt-4o"):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0, # increase temperature
            max_tokens=1024, # 512?
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # messages = [{"role": "user", "content": prompt}]
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=messages,
        #     temperature=0,
        # )
        return response.choices[0].message.content

    for idx in range(20):
        with open(f"{folder_name.replace('answers', 'prompts')}/{idx}.txt", "r") as f:
            prompt = f.readlines()
        prompt = "".join(prompt)
        # print(prompt)
        response = get_completion(prompt=prompt)
        # response = """
        # Answer: 113, 143
        # """
        print(response)
        answer_idx = response.find("Answer: ")
        answer = response[answer_idx+len("Answer: "):]
        if "," not in answer or answer_idx == -1:
            x, y = -1, -1
        else:
            x, y = answer.split(",")
            if "." in y:
                y = y[:-1]
            x = int(x.strip())
            y = int(y.strip())
        with open(f"{folder_name}/{idx}.json", "w") as f:
            json.dump({"x": x, "y": y}, f)
        with open(f"{folder_name}/{idx}.txt", "w", encoding='utf-8') as f:
            f.write(response)
        print(x, y)
# find the answer part
