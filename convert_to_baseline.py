import os
import pickle
import cv2
import numpy as np
for idx in range(3, 8):
    prompt_folder_name = f"if_prompts_{idx}_ours"
    examples_folder_name = f"if_examples_{idx}_ours"
    os.makedirs(f"data/{prompt_folder_name}_vanilla", exist_ok=True)
    os.makedirs(f"data/{examples_folder_name}_vanilla", exist_ok=True)
    for file_idx in range(20):
        with open(f"data/{examples_folder_name}/{file_idx}_info.pkl", "rb") as f:
            info = pickle.load(f)
        # need to delete ambiguous object from prompt
        # remove it from _mask
        obj_name = info["gt_info"]["correct_name"]
        for word, _, _, _, _, _ in info["ref_lists"]:
            words = word.split(" ")
            if len(words) == 1 or words[1] == "thing":
                overlap_word = words[0]
                break
        # print(overlap_word)
        # print(obj_name)
        for word in info["info"]:
            if word != obj_name and overlap_word in word:
                # need to remove this!
                # print("******")
                # print(word, obj_name)
                target_word = word
                target_bbox = info["info"][word]
                break
        # remove from the mask

        # remove from the prompt
        with open(f"data/{prompt_folder_name}/{file_idx}.txt", "r") as f:
            prompt = f.readlines()
        new_prompt = ""
        for p in prompt:
            if target_word in p:
                continue
            else:
                new_prompt += p
        # print(new_prompt)
        img = cv2.imread(f"data/{examples_folder_name}/{file_idx}_mask.png", cv2.IMREAD_GRAYSCALE)
        prev_len = np.unique(img).shape[0]
        x, y, w, h = target_bbox
        img[y:y+h, x:x+w] = 0 # no overlap now!
        assert np.unique(img).shape[0] + 1 == prev_len
        
        cv2.imwrite(f"data/{examples_folder_name}_vanilla/{file_idx}_mask.png", img)
        with open(f"data/{prompt_folder_name}_vanilla/{file_idx}.txt", "w") as f:
            f.writelines(new_prompt)
        with open(f"data/{examples_folder_name}_vanilla/{file_idx}_info.pkl", "wb") as f:
            pickle.dump(info, f)
        
        # print(target_word)   
        # print(info["gt_info"]["important_object"])
        # print(info["ref_lists"])
        # print(info["info"])
        # exit()