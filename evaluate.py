import json
import pickle
import numpy as np
import cv2
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_relation", type=int)
parser.add_argument("--important_object_first", action="store_true")
parser.add_argument("--ours", action="store_true")
parser.add_argument("--rewrite", action="store_true")
parser.add_argument("--logical", action="store_true")
parser.add_argument("--combined", action="store_true")
parser.add_argument("--vanilla", action="store_true")

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
    scores = [["Correct?", "Due to things?"]]
    # check whether the given relationship satisfies the whole relations
    for i in range(20):
        correct = True
        with open(f"{folder_name}/{i}.json", "r") as f:
            data = json.load(f)
        with open(f"{folder_name.replace('answers', 'examples')}/{i}_info.pkl", "rb") as f:
            raw_info = pickle.load(f)
        x, y = data["x"], data["y"]
        mask = cv2.imread(f"{folder_name.replace('answers', 'examples')}/{i}_mask.png", cv2.IMREAD_GRAYSCALE)
        important_object = raw_info["gt_info"]["important_object"][0]
        _, _, _, tar_w, tar_h = raw_info["gt_info"]["target_position"]
        # print(mask[y, x])
        if np.any(mask[y:y+tar_h, x:x+tar_w]) != 0:
            correct = False
            print("overlap")
        if x == -1 or y == -1:
            correct = False
            print("no valid output")
        wrong_word = []
        # if correct:
        for word, x0, y0, w, h, direction in raw_info["ref_lists"]:
            if direction == "left":
                if not x < (x0 + w):
                    # then wrong
                    print(word, direction, x, x0)
                    correct = False
                    wrong_word.append(word)
                    # break
            elif direction == "right":
                if not (x + tar_w) > (x0):
                    correct = False
                    print(word, direction, x, x0, w, x0 + w)
                    wrong_word.append(word)
                    # break
            elif direction == "above":
                if not y < (y0 + h):
                    correct = False
                    print(word, direction, y, y0)
                    wrong_word.append(word)
                    # break
            elif direction == "below":
                if not (y + tar_h) > (y0):
                    correct = False
                    print(word, direction, y, y0, h, y0+h)
                    wrong_word.append(word)
                    # break
        print(correct)
        # a = input(wrong_word)
        # check whether it satisfies 'thing'
        wrong_thing = False
        for word in wrong_word:
            if 'thing' in word or " " not in word or word == important_object:
                wrong_thing = True
                break
        if x == -1 or y == -1:
            wrong_thing = True
        if not wrong_thing:
            print(wrong_word)
            print(important_object)
        
        scores.append([correct, wrong_thing])
    print(scores)
    with open(f"result_log/{folder_name.replace('data/', '')}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(scores)
    # tends to overlap with other objects, seems like fail to do backward inference
    # we can track the target object by reading the prompt output!
    # shuffle degrades performance a lot!
    # maybe 6 is too long for them?