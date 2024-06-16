import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_relation", type=int)
parser.add_argument("--important_object_first", action="store_true")
parser.add_argument("--ours", action="store_true")
parser.add_argument("--logical", action="store_true")
parser.add_argument("--combined", action="store_true")
parser.add_argument("--rewrite", action="store_true")

args = parser.parse_args()

scene_w, scene_h = 200, 200
min_size = 10
max_size = 30
max_relations = args.num_relation # 6
max_objects = args.num_relation + 4 # 10
important_object_first = args.important_object_first
if args.important_object_first:
    folder_name = f"data/if_examples_{args.num_relation}"
else:
    folder_name = f"data/examples_{args.num_relation}"
if args.ours:
    folder_name += "_ours"
if args.logical:
    folder_name += "_logical"
if args.combined:
    folder_name += "_combined"
if args.rewrite:
    folder_name += "_rewrite"
os.makedirs(folder_name, exist_ok=True)
os.makedirs(folder_name.replace("examples", "prompts"), exist_ok=True)

def get_left_negate(ref_bbox, w, h):
    scene = np.zeros((scene_w, scene_h))
    x0, _, ref_w, _ = ref_bbox
    x = x0 + ref_w
    if x > scene_w - w:
        return scene
    else:
        scene[:scene_h-h, x:scene_w-w] = 1
    return scene
def get_right_negate(ref_bbox, w, h):
    scene = np.zeros((scene_w, scene_h))
    x0, _, ref_w, _ = ref_bbox
    x = x0 - w - ref_w
    if x < 0:
        return scene
    else:
        scene[:scene_h-h, 0:x] = 1
    return scene
def get_above_negate(ref_bbox, w, h):
    # below
    scene = np.zeros((scene_w, scene_h))
    _, y0, _, ref_h = ref_bbox
    y = y0 + ref_h
    if y > scene_h - h:
        return scene
    else:
        scene[y:scene_h-h, 0:scene_w-w] = 1
    return scene
def get_below_negate(ref_bbox, w, h):
    # above
    scene = np.zeros((scene_w, scene_h))
    _, y0, _, ref_h = ref_bbox
    y = y0 - h - ref_h # this could be negative...
    if y < 0:
        return scene
    else:
        scene[0:y, 0:scene_w-w] = 1
    return scene
def get_near_negate(ref_bbox, w, h):
    # near
    scene = np.zeros((scene_w, scene_h))
    x0, y0, ref_w, ref_h = ref_bbox
    ctx_x = int(round(x0 + ref_w/2))
    ctx_y = int(round(y0 + ref_h/2))
    yy, xx = np.mgrid[:scene_h, :scene_w]
    circle = (xx - ctx_x) ** 2 + (yy - ctx_y) ** 2
    distance_threshold = 2 * max(ref_w, ref_h)
    scene = np.where(circle < distance_threshold*distance_threshold, 1, 0)
    return scene
def get_far_negate(ref_bbox, w, h):
    # far
    scene = np.zeros((scene_w, scene_h))
    x0, y0, ref_w, ref_h = ref_bbox
    ctx_x = int(round(x0 + ref_w/2))
    ctx_y = int(round(y0 + ref_h/2))
    yy, xx = np.mgrid[:scene_h, :scene_w]
    circle = (xx - ctx_x) ** 2 + (yy - ctx_y) ** 2
    distance_threshold = 2 * max(ref_w, ref_h)
    scene = np.where(circle > distance_threshold*distance_threshold, 1, 0)
    return scene

ori_colors = ["red", "orange", "yellow" ,"green", "blue", "purple", "pink", "white", "black", "brown"]
ori_shapes = ["ball", "box", "button", "flower", "pen", "toy", "eraser", "pencil"]
relations = ["left", "right", "above", "below", "near", "far"]
relations_to_negate = {"left": get_left_negate, "right": get_right_negate, "above": get_above_negate, "below": get_below_negate, "near": get_near_negate, "far": get_far_negate}
assert max_relations + 2 <= max_objects
random.seed(0)

if args.ours:
    prompt_template = f"""Find space in the given scene that satisfies the given instruction. The scene is described as a list with an object name and bounding box (x0, y0, w, h). First, re-order the given instruction to make the task as easy as possible while keeping the meaning same. Then, always write only one sentence to describe the reasoning process based on the new instruction and return the answer in the format "Answer: x, y".
Scene size: {scene_w} x {scene_h} (w, h)
Scene
"""
elif args.rewrite:
    prompt_template = f"""Find space in the given scene that satisfies the given instruction. The scene is described as a list with an object name and bounding box (x0, y0, w, h). First, re-write the given instruction to make the task as easy as possible while keeping the meaning same. Then, always write only one sentence to describe the reasoning process based on the new instruction and return the answer in the format "Answer: x, y".
Scene size: {scene_w} x {scene_h} (w, h)
Scene
"""
elif args.logical:
    prompt_template = f"""Find space in the given scene that satisfies the given instruction. The scene is described as a list with an object name and bounding box (x0, y0, w, h). First, convert the given instruction into first order logic. Then, always write only one sentence to describe the reasoning process based on the generated logic and return the answer in the format "Answer: x, y".
Scene size: {scene_w} x {scene_h} (w, h)
Scene
"""
elif args.combined:
    prompt_template = f"""Find space in the given scene that satisfies the given instruction. The scene is described as a list with an object name and bounding box (x0, y0, w, h). First, re-order the given instruction to make the task as easy as possible while keeping the meaning same. Second, convert the re-ordered instruction into first order logic. Then, always write only one sentence to describe the reasoning process based on the generated logic and return the answer in the format "Answer: x, y".
Scene size: {scene_w} x {scene_h} (w, h)
Scene
"""
else:
    prompt_template = f"""Find space in the given scene that satisfies the given instruction. The scene is described as a list with an object name and bounding box (x0, y0, w, h). Always write only one sentence to describe the reasoning process and return the answer in the format "Answer: x, y".
Scene size: {scene_w} x {scene_h} (w, h)
Scene
"""
# - blue cube: 30, 60, 20, 40
# - red cylinder: 60, 30, 30, 50
# - green ball: 150, 150, 10, 10
# - yellow flower: 10, 0, 10, 10
# Instruction: Find space to put the green ball at the left of red cylinder, the above of the blue thing, and close to the yellow flower."""

for img_idx in range(20):
    valid = False
    while not valid:
        colors = copy.deepcopy(ori_colors)
        shapes = copy.deepcopy(ori_shapes)
        # print(scene_w)
        scene = np.zeros((scene_w, scene_h), dtype=np.uint8)
        # choose the rough target region
        gt_infos = {}
        infos = {}
        ref_lists = []
        color = random.choice(colors)
        shape = random.choice(shapes)
        word = f"{color} {shape}"
        # choose random number for x0, y0, w, h
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x0 = random.randint(0, scene_w - w)
        y0 = random.randint(0, scene_h - h)
        gt_infos["target_position"] = [word, x0, y0, w, h]
        # infos[word] = (x0, y0, w, h) # do not add to info! cause this is the target position
        scene[y0:y0+h, x0:x0+w] = 1

        # choose relationship
        color = random.choice(colors)
        shape = random.choice(shapes)
        word = f"{color} {shape}"
        # choose random number for x0, y0, w, h
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        direction = random.choice(relations[:4])
        # then negate it!
        negate_scene = relations_to_negate[direction](gt_infos["target_position"][1:], w, h)
        # find the part that is not in the scene and at the same time negate_scene == 1
        ys, xs = np.where((scene == 0) & (negate_scene == 1))
        if len(ys) == 0:
            # no place to put!
            continue
        idx = random.choice(list(range(len(ys))))
        x0 = xs[idx]
        y0 = ys[idx]
        gt_infos["important_object"] = [word, x0, y0, w, h, direction]
        ref_lists.append([word, x0, y0, w, h, direction])
        infos[word] = (x0, y0, w, h)
        scene[y0:y0+h, x0:x0+w] = 2

        # then choose the overlapped ones!!!!
        overlap_color = random.choice(list(range(2)))
        base_direction = direction[:]
        if overlap_color == 0:
            common_color = random.choice(colors)
            while common_color == color:
                common_color = random.choice(colors)
            word = f"{common_color} thing"
            colors.remove(common_color)
        else:
            common_shape = random.choice(shapes)
            while common_shape == shape:
                common_shape = random.choice(shapes)
            word = f"{common_shape}"
            shapes.remove(common_shape)
        # then put the object at the left side of the target object
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        if base_direction == "left":
            direction = "right"
        elif base_direction == "right":
            direction = "left"
        elif base_direction == "above":
            direction = "below"
        else:
            direction = "above"
        # then negate it!
        negate_scene = relations_to_negate[direction](gt_infos["target_position"][1:], w, h)
        # find the part that is not in the scene and at the same time negate_scene == 1
        tmp_scene = np.where(scene == 0, 0, 1).astype(np.uint8)
        tmp_scene = cv2.dilate(tmp_scene, np.ones((max(w, h), max(w, h)), dtype=np.uint8), iterations=1)
        ys, xs = np.where((tmp_scene == 0) & (negate_scene == 1))
        if len(ys) == 0:
            # no place to put!
            continue
        idx = random.choice(list(range(len(ys))))
        x0 = xs[idx]
        y0 = ys[idx]
        gt_infos["common_object"] = [word, x0, y0, w, h, direction]
        ref_lists.append([word, x0, y0, w, h, direction])
            
        if overlap_color == 0:
            shape = random.choice(shapes)
            real_word = f"{common_color} {shape}"
        else:
            color = random.choice(colors)
            real_word = f"{color} {common_shape}"
        infos[real_word] = (x0, y0, w, h)
        scene[y0:y0+h, x0:x0+w] = 3
        gt_infos["correct_name"] = real_word

        # then, now put the distractor object at the base_direction of the imporatnat object
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        negate_scene = relations_to_negate[base_direction](gt_infos["important_object"][1:5], w, h)
        tmp_scene = np.where(scene == 0, 0, 1).astype(np.uint8)
        tmp_scene = cv2.dilate(tmp_scene, np.ones((max(w, h), max(w, h)), dtype=np.uint8), iterations=1)
        ys, xs = np.where((tmp_scene == 0) & (negate_scene == 1))
        if len(ys) == 0:
            # no place to put!
            continue
        idx = random.choice(list(range(len(ys))))
        x0 = xs[idx]
        y0 = ys[idx]
        # this is just negater
        if overlap_color == 0:
            shape = random.choice(shapes)
            real_word = f"{common_color} {shape}"
        else:
            color = random.choice(colors)
            real_word = f"{color} {common_shape}"
        # if img_idx == 1:
        #     print(base_direction, img_idx, real_word)
        #     _ = input("")
        infos[real_word] = (x0, y0, w, h)
        scene[y0:y0+h, x0:x0+w] = 4

        # then add a random number of ref object
        relation_length = max_relations # random.randint(2, max_relations)
        for obj_idx in range(relation_length - 2):
            # choose relationship
            color = random.choice(colors)
            shape = random.choice(shapes)
            word = f"{color} {shape}"
            # choose random number for x0, y0, w, h
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            direction = random.choice(relations)
            # then negate it!
            negate_scene = relations_to_negate[direction](gt_infos["target_position"][1:], w, h)
            # find the part that is not in the scene and at the same time negate_scene == 1
            tmp_scene = np.where(scene == 0, 0, 1).astype(np.uint8)
            tmp_scene = cv2.dilate(tmp_scene, np.ones((max(w, h), max(w, h)), dtype=np.uint8), iterations=1)
            ys, xs = np.where((tmp_scene == 0) & (negate_scene == 1))
            if len(ys) == 0:
                # no place to put!
                break
            idx = random.choice(list(range(len(ys))))
            x0 = xs[idx]
            y0 = ys[idx]
            ref_lists.append([word, x0, y0, w, h, direction])
            if word in infos:
                break
            infos[word] = (x0, y0, w, h)
            scene[y0:y0+h, x0:x0+w] = 5 + obj_idx
            # print("a", 5 + obj_idx)
        if len(ref_lists) != relation_length:
            continue
        # put the target object in the proper position
        word, _, _, w, h = gt_infos["target_position"]
        tmp_scene = np.where(scene == 0, 0, 1).astype(np.uint8)
        tmp_scene = cv2.dilate(tmp_scene, np.ones((max(w, h), max(w, h)), dtype=np.uint8), iterations=1)
        ys, xs = np.where((tmp_scene == 0) & (negate_scene == 1))
        if len(ys) == 0:
            # no place to put!
            continue
        idx = random.choice(list(range(len(ys))))
        x0 = xs[idx]
        y0 = ys[idx]
        infos[word] = (x0, y0, w, h)
        scene[y0:y0+h, x0:x0+w] = relation_length + 2 + 1
        # and add a random number of distractor
        num_objects = random.randint(relation_length + 2, max_objects)
        for obj_idx in range(num_objects - 2 - relation_length):
            # choose relationship
            color = random.choice(colors)
            shape = random.choice(shapes)
            word = f"{color} {shape}"
            # choose random number for x0, y0, w, h
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            tmp_scene = np.where(scene == 0, 0, 1).astype(np.uint8)
            tmp_scene = cv2.dilate(tmp_scene, np.ones((max(w, h), max(w, h)), dtype=np.uint8), iterations=1)
            ys, xs = np.where((tmp_scene == 0))
            if len(ys) == 0:
                # no place to put!
                break
            idx = random.choice(list(range(len(ys))))
            x0 = xs[idx]
            y0 = ys[idx]
            if word in infos:
                break
            infos[word] = (x0, y0, w, h)
            scene[y0:y0+h, x0:x0+w] = relation_length + 2 + obj_idx + 2
            # print("b", relation_length + 2 + obj_idx)
            # print("b", np.unique(scene))
        if len(infos) != num_objects:
            continue
        # then generate sentence!
        # need to shffle well to make sure to satisfy condtion
        idxs = list(range(relation_length))
        random_idx = random.randint(1, relation_length - 1)
        if important_object_first:
            imp_idx = 0
            amb_idx = random_idx
        else:
            amb_idx = 0 # random.randint(0, relation_length - 2)
            imp_idx = random_idx
        # print(amb_idx)
        key_map = {imp_idx: 0, amb_idx: 1}
        rest_idxs = list(set(idxs) - set([imp_idx, amb_idx]))
        random.shuffle(rest_idxs)
        for i, k in enumerate(rest_idxs):
            key_map[k] = i+2
        ori_ref_lists = copy.deepcopy(ref_lists)
        ref_lists = [ori_ref_lists[key_map[i]] for i in range(relation_length)]
        instruction = f'place the {gt_infos["target_position"][0]} at the '
        for ref_idx, (word, _, _, _, _, direction) in enumerate(ref_lists):
            if direction == "near":
                if ref_idx == 0:
                    instruction = instruction[:-7]
                else:
                    instruction = instruction[:-4]
                instruction += f"{direction} to the {word} and the "
            elif direction == "far":
                if ref_idx == 0:
                    instruction = instruction[:-7]
                else:
                    instruction = instruction[:-4]
                instruction += f"{direction} from the {word} and the "
            else:
                instruction += f"{direction} of the {word} and the "
        instruction = instruction[:-8]
        # print(instruction)
        # print(gt_infos["important_object"])
        plt.figure()
        plt.imshow(scene)
        plt.colorbar()
        # print(len(infos))
        # print(np.unique(scene))
        for idx, word in enumerate(infos.keys()):
            ys, xs = np.where(scene == idx + 2)
            plt.text(min(xs), min(ys), word, color='gray', fontsize=12)
            # print(word, min(xs), min(ys))
        plt.title(instruction, wrap=True)
        plt.savefig(f"{folder_name}/{img_idx}.png")
        scene = np.where((scene == 1) | (scene == relation_length + 2 + 1), 0, scene)
        cv2.imwrite(f"{folder_name}/{img_idx}_mask.png", scene)
        total_info = {}
        total_info["info"] = infos
        total_info["gt_info"] = gt_infos
        total_info["ref_lists"] = ref_lists
        total_info["instruction"] = instruction
        # print(total_info)
        with open(f"{folder_name}/{img_idx}_info.pkl", "wb") as f:
            pickle.dump(total_info, f)
        valid = True
        prompt = prompt_template[:]
        word_list = list(infos.keys())
        random.shuffle(word_list)
        for word in word_list:
            x0, y0, w, h = infos[word]
            prompt += f"- {word}: {x0}, {y0}, {w}, {h}\n"
        prompt += f"Instruction: {instruction}"
        with open(f"{folder_name.replace('examples', 'prompts')}/{img_idx}.txt", "w") as f:
            f.writelines(prompt)
            





    