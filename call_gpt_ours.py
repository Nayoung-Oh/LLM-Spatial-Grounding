from openai import OpenAI
from dotenv import load_dotenv
import json
# pip install python-dotenv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_relation", type=int)
parser.add_argument("--important_object_first", action="store_true")
args = parser.parse_args()
important_object_first = args.important_object_first
if args.important_object_first:
    folder_name = f"data/if_answers_{args.num_relation}"
else:
    folder_name = f"data/answers_{args.num_relation}"
# if args.ours:
folder_name += "_ours"
os.makedirs(folder_name, exist_ok=True)
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
def get_completion(prompt, model="gpt-4o"):
    global logs
    messages = [{"role": "user", "content": prompt}]
    logs += prompt + "\n"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2, # increase temperature
        max_tokens=512,
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
    ans = response.choices[0].message.content
    logs += (ans + "\n*****************************\n")
    return ans

rephrase_prompt = f'''Write the given instruction again in just different order. (eg. place A at left of B and right of C -> place A at right of C and left of B)
Answer in the format "Instruction: ~"\n'''
# print(folder_name)
ori_prompt = 'Find space in the given scene that satisfies the given instruction. The scene is described as a list with an object name and bounding box (x0, y0, w, h). Always write only one sentence to describe the reasoning process and return the answer in the format "Answer: x, y".'
verify_prompt = f'''You need to check whether the given statement is true or false. The scene is described as a list with an object name and bounding box (x0, y0, w, h). Always write only one sentence to describe the reasoning process and return the answer in the format "Answer: boolean" as a last sentence. boolean should be true or false'''
for idx in range(20):
    logs = ""
    with open(f"{folder_name.replace('_ours', '').replace('answers', '/prompts')}/{idx}.txt", "r") as f:
        prompt = f.readlines()
    prompt = "".join(prompt)
    i_id = prompt.find("Instruction: ")
    instruction = prompt[i_id:]
    front_prompt = prompt[:i_id].replace(ori_prompt, verify_prompt)
    stop = False
    for _ in range(3):
        if stop:
            break
        ### ALICE ###
        print(prompt)
        response = get_completion(prompt=prompt)
        # then ask the other session, just to rephrase it
        
        # response = """
        # Answer: 113, 143
        # """
        print(response)
        answer_idx = response.find("Answer: ")
        answer = response[answer_idx+len("Answer: "):].strip()
        if "," not in answer or answer_idx == -1:
            x, y = -1, -1
        else:
            x, y = answer.split(",")
            if "." in y:
                y = y[:-1]
            x = int(x.strip())
            y = int(y.strip())
        
        ### CHALLY ###
        # check whether the given value satisfy the sentence
        # ins_idx = instruction.find(": ")
        instruction_part = instruction + " So, I plcae it to " + f"point: {x}, {y}"
        
        new_new_prompt = front_prompt + instruction_part
        print(new_new_prompt)
        new_new_response = get_completion(prompt=new_new_prompt)
    #     new_new_response = """The pink eraser is not above the pink pencil, as its y-coordinate (161) is greater than the pink pencil's y-coordinate (172).

    # Answer: false"""
        print("**************")
        print(new_new_response)
        answer_idx = new_new_response.find("Answer: ")
        resp = new_new_response[answer_idx+len("Answer: "):].strip()
        _ = input(resp)
        print("**************")
        # resp = input(resp)
        if resp == 'true' or resp == "True":
            stop = True
            print("ENNNNNNNNNNNNNNNNNND", idx)
        elif resp == 'false' or resp == "False":
            stop = False
        else:
            print("Weird", resp)
            stop = False
        if not stop:
            ### BOB ### -> maybe replace latter
            new_prompt = rephrase_prompt + instruction
            print(new_prompt)
            new_response = get_completion(prompt=new_prompt)
            # new_response = "Instruction: place the pink eraser above the pink pencil and to the right of the pink pen and to the left of the blue thing."
            prompt = prompt.replace(instruction, new_response)
            instruction = new_response
    # exit()
    
    with open(f"{folder_name}/{idx}.json", "w") as f:
        json.dump({"x": x, "y": y}, f)
    with open(f"{folder_name}/{idx}.txt", "w") as f:
        f.write(logs)
    print(x, y)
# find the answer part
