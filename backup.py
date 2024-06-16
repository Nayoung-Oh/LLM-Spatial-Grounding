import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

scene_w, scene_h = 200, 200
min_size = 10
max_size = 30
colors = ["red", "orange", "yellow" ,"green", "blue", "purple", "pink", "white", "black", "brown"]
shapes = ["ball", "box", "button", "flower", "pen", "toy", "eraser", "pencil"]
relations = ["left", "right", "above", "below", "near", "far"]
random.seed(0)
for i in range(10):
    valid = False
    while not valid:
        scene = np.zeros((scene_w, scene_h))
        infos = {}
        gt_infos = {}
        ref_lists = []
        sentence = "Place the "
        # put one object to move!
        color = random.choice(colors)
        shape = random.choice(shapes)
        word = f"{color} {shape}"
        # choose random number for x0, y0, w, h
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x0 = random.randint(0, scene_w - w)
        y0 = random.randint(0, scene_h - h)
        gt_infos["target_object"] = (word, x0, y0, w, h)
        infos[word] = (x0, y0, w, h)
        sentence += word
        scene[y0:y0+h, x0:x0+w] = 1

        # put the other object as the middle one
        # put one object to move!
        color = random.choice(colors)
        shape = random.choice(shapes)
        word = f"{color} {shape}"
        # choose random number for x0, y0, w, h
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x0 = random.randint(0, scene_w - w)
        y0 = random.randint(0, scene_h - h)
        direction = random.choice(relations[:4])
        ref_lists.append((word, x0, y0, w, h, direction))
        gt_infos["important_object"] = (word, x0, y0, w, h, direction)
        infos[word] = (x0, y0, w, h)
        if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
            continue
        scene[y0:y0+h, x0:x0+w] = 1

        # let's first put the ambiguous object
        overlap_color = random.choice(list(range(2)))
        if overlap_color == 0:
            common_color = random.choice(colors)
            while common_color == color:
                common_color = random.choice(colors)
            word = f"{common_color} thing"
            # choose two shape
            if direction == "left":
                # then, put the object at the left side of it
                _, _, _, obj_w, _ = infos["target_object"]
                _, im_x, _, _, _ = infos["important_object"]
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x0 = random.randint(0, im_x - w - obj_w)
                y0 = random.randint(0, scene_h - h)
                if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
                    continue
                scene[y0:y0+h, x0:x0+w] = 1
                # this one is valid!
                gt_infos["common_object"] = (word, x0, y0, w, h, "right")
                shape = random.choice(shapes)
                infos[f"{common_color} {shape}"] = (x0, y0, w, h)
                ref_lists.append((word, x0, y0, w, h, "right"))
                # then choose the object at the reverse side
                _, _, _, obj_w, obj_h = infos["target_object"]
                _, im_x, _, im_w, _ = infos["important_object"]
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x0 = random.randint(im_x + im_w, scene_w - w)
                y0 = random.randint(0, scene_h - h)
                if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
                    continue
                scene[y0:y0+h, x0:x0+w] = 1
                prev_shape = shape[:]
                shape = random.choice(shapes)
                while prev_shape == shape:
                    shape = random.choice(shapes)
                infos[f"{common_color} {shape}"] = (x0, y0, w, h)
            elif direction == "right":
                # then, put the object at the right side of it
                _, _, _, obj_w, _ = infos["target_object"]
                _, im_x, _, im_w, _ = infos["important_object"]
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x0 = random.randint(im_x + im_w + obj_w, scene_w - w)
                y0 = random.randint(0, scene_h - h)
                if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
                    continue
                scene[y0:y0+h, x0:x0+w] = 1
                # this one is valid!
                gt_infos["common_object"] = (word, x0, y0, w, h, "left")
                shape = random.choice(shapes)
                infos[f"{common_color} {shape}"] = (x0, y0, w, h)
                ref_lists.append((word, x0, y0, w, h, "left"))
                # then choose the object at the reverse side
                _, _, _, obj_w, obj_h = infos["target_object"]
                _, im_x, _, im_w, _ = infos["important_object"]
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x0 = random.randint(0, im_x - w)
                y0 = random.randint(0, scene_h - h)
                if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
                    continue
                scene[y0:y0+h, x0:x0+w] = 1
                prev_shape = shape[:]
                shape = random.choice(shapes)
                while prev_shape == shape:
                    shape = random.choice(shapes)
                infos[f"{common_color} {shape}"] = (x0, y0, w, h)
            elif direction == "right":
                # then, put the object at the right side of it
                _, _, _, obj_w, _ = infos["target_object"]
                _, im_x, _, im_w, _ = infos["important_object"]
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x0 = random.randint(im_x + im_w + obj_w, scene_w - w)
                y0 = random.randint(0, scene_h - h)
                if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
                    continue
                scene[y0:y0+h, x0:x0+w] = 1
                # this one is valid!
                gt_infos["common_object"] = (word, x0, y0, w, h, "left")
                shape = random.choice(shapes)
                infos[f"{common_color} {shape}"] = (x0, y0, w, h)
                ref_lists.append((word, x0, y0, w, h, "left"))
                # then choose the object at the reverse side
                _, _, _, obj_w, obj_h = infos["target_object"]
                _, im_x, _, im_w, _ = infos["important_object"]
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x0 = random.randint(0, im_x - w)
                y0 = random.randint(0, scene_h - h)
                if np.any(scene[y0:y0+h, x0:x0+w]) == 1:
                    continue
                scene[y0:y0+h, x0:x0+w] = 1
                prev_shape = shape[:]
                shape = random.choice(shapes)
                while prev_shape == shape:
                    shape = random.choice(shapes)
                infos[f"{common_color} {shape}"] = (x0, y0, w, h)