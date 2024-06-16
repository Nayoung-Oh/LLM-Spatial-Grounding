import os
import json
re_evaluate = []
for folder in sorted(os.listdir("data")):
    if "answer" not in folder:
        continue
    print(folder)
    for i in range(20):
        with open(os.path.join("data", folder, f"{i}.json"), "r") as f:
            value = json.load(f)
        if value["x"] == -1 or value["y"] == -1:
            print(value, folder, i)
            with open(os.path.join("data", folder, f"{i}.txt"), "r") as f:
                answer = f.readlines()
            print(answer)
            x = int(input("new x?"))
            y = int(input("new_y?"))
            value = {"x":x, "y": y}
            if x != -1 and y != -1:
                with open(os.path.join("data", folder, f"{i}.json"), "w") as f:
                    json.dump(value, f)
                re_evaluate.append(folder)
            else:
                print("continue~~")
            # exit()
print(re_evaluate)
    