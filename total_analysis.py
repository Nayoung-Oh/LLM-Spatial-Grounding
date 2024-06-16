import os
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import numpy as np
result_accuracy = {"if": [[], []], "if_ours": [[], []], "if_logical": [[], []], "if_combined": [[], []], "if_vanilla": [[], []], "if_ours_vanilla": [[], []], "if_rewrite": [[], []],
                   "la": [[], []], "la_ours": [[], []], "la_logical": [[], []], "la_combined": [[], []], "la_vanilla": [[], []], "la_ours_vanilla": [[], []], "la_rewrite": [[], []]}
result_ths = {"if": [[], []], "if_ours": [[], []], "if_logical": [[], []], "if_combined": [[], []], "if_vanilla": [[], []], "if_ours_vanilla": [[], []], "if_rewrite": [[], []],
              "la": [[], []], "la_ours": [[], []], "la_logical": [[], []], "la_combined": [[], []], "la_vanilla": [[], []], "la_ours_vanilla": [[], []], "la_rewrite": [[], []]}
color_list = {"if": '#ff0000', "if_ours": '#ff5580', "if_logical": '#ff6500', "if_vanilla": "#888888", "if_combined": "#33ff00", "if_ours_vanilla": "#473891", "if_rewrite": "#008783",
              "la": '#0000ff', "la_ours": "#874CCC", "la_logical": "#C5FF95", "la_combined": "#ffff00", "la_vanilla": "#000000", "la_ours_vanilla": "#091236", "la_rewrite": "#567821"}
# if_accuracy_tot = [[], []]
# accuracy_tot = [[], []]

# if_ths_tot = [[], []]
# ths_tot = [[], []]

# if_ths_tot_2 = [[], []]
# ths_tot_2 = [[], []]
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
for file_name in sorted(os.listdir("result_log")):
    accuracy = []
    things_impact = []
    with open(os.path.join("result_log", file_name), "r") as f:
        reader = csv.reader(f)
        for i, data in enumerate(reader):
            if i == 0:
                continue
            co, th = data
            accuracy.append(1 if co == "True" else 0)
            if co == "False":
                things_impact.append(1 if th == "True" else 0)
    acc = sum(accuracy) / len(accuracy)
    if len(things_impact) > 0:
        ths = sum(things_impact) / len(things_impact)
        ths_2 = sum(things_impact) / len(accuracy)
    else:
        ths = 0
        ths_2 = 0
    print(acc, ths)
    num = int(file_name.replace("_rewrite", "").replace("_ours", "").replace("_logical", "").replace("_combined", "").replace("_vanilla", "").split("_")[-1].replace(".csv", ""))
    
    name = ""
    if "if_" in file_name:
        name = "if"
    else:
        name = 'la'
    if "ours" in file_name:
        name += "_ours"
    if "logical" in file_name:
        name += "_logical"
    if "combined" in file_name:
        name += "_combined"
    if "vanilla" in file_name:
        name += "_vanilla"
    if "rewrite" in file_name:
        name += "_rewrite"
    result_accuracy[name][0].append(num)
    result_accuracy[name][1].append(acc)
    result_ths[name][0].append(num)
    result_ths[name][1].append(ths)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Total peformance", wrap=True)
for label, (num, acc) in result_accuracy.items():
    plt.plot(num, acc, 'o-', color=color_list[label], label=label)
# plt.plot(if_accuracy_tot[0], if_accuracy_tot[1], 'ro-', label="important first")
# plt.plot(accuracy_tot[0], accuracy_tot[1], 'bo-', label="important latter")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 10.0)
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Wrong answer due to wrong ref\n(in wrong anwer)", wrap=True)
for label, (num, acc) in result_ths.items():
    plt.plot(num, acc, 'o-', color=color_list[label], label=label)
# plt.plot(if_ths_tot[0], if_ths_tot[1], 'ro-', label="important first")
# plt.plot(ths_tot[0], ths_tot[1], 'bo-', label="important latter")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 10.0)
plt.legend()
# plt.subplot(1, 3, 3)
# plt.title("Wrong answer due to wrong ref\n(in total anwer)", wrap=True)
# plt.plot(if_ths_tot_2[0], if_ths_tot_2[1], 'ro-', label="important first")
# plt.plot(ths_tot_2[0], ths_tot_2[1], 'bo-', label="important latter")
# plt.ylim(0.0, 1.0)
# plt.legend()
plt.savefig("result.png")
plt.show()

plt.figure(figsize=(15, 4))
tmp = list(result_accuracy.values())
tmp = np.array(tmp)
a = tmp[:, 1, :] # 6 by 5
a = np.mean(a, axis=1)
a = np.round(a, decimals=3)
names = list(result_accuracy.keys())
color = np.array(["g"] * len(names))
print(a, color)
color[a < 0.6] = "r"
color[(a < 0.7) & (a >= 0.6)] = "y"
color[(a < 0.8) & (a >= 0.7)] = "g"
color[a > 0.8] = "b"
plt.bar(np.arange(len(names)), a, color=color)
addlabels(np.arange(len(names)), a)
plt.xticks(np.arange(len(names)), names)
plt.savefig("bar_plot.png")
plt.show()


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Total peformance", wrap=True)
tmp = list(result_accuracy.values())
tmp = np.array(tmp)
a = tmp[:, 1, :] # 6 by 5
a = a.reshape(2, 7, 5)
a = np.mean(a, axis=0)
for i, name in enumerate(["original", "ours", "logical", "combined", "vanilla", "vanilla_ours", "rewrite"]):
    plt.plot(num, a[i, :], 'o-', label=name)
# for label, (num, acc) in result_accuracy.items():
#     plt.plot(num, acc, 'o-', color=color_list[label], label=label)
# # plt.plot(if_accuracy_tot[0], if_accuracy_tot[1], 'ro-', label="important first")
# # plt.plot(accuracy_tot[0], accuracy_tot[1], 'bo-', label="important latter")
plt.ylim(0.0, 1.0)
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Wrong answer due to wrong ref\n(in wrong anwer)", wrap=True)
tmp = list(result_ths.values())
tmp = np.array(tmp)
a = tmp[:, 1, :] # 6 by 5
a = a.reshape(2, 7, 5)
a = np.mean(a, axis=0)
for i, name in enumerate(["original", "ours", "logical", "combined", "vanilla", "vanilla_ours", "rewrite"]):
    plt.plot(num, a[i, :], 'o-', label=name)
# for label, (num, acc) in result_ths.items():
#     plt.plot(num, acc, 'o-', color=color_list[label], label=label)
# # plt.plot(if_ths_tot[0], if_ths_tot[1], 'ro-', label="important first")
# # plt.plot(ths_tot[0], ths_tot[1], 'bo-', label="important latter")
plt.ylim(0.0, 1.0)
plt.legend()
# # plt.subplot(1, 3, 3)
# # plt.title("Wrong answer due to wrong ref\n(in total anwer)", wrap=True)
# # plt.plot(if_ths_tot_2[0], if_ths_tot_2[1], 'ro-', label="important first")
# # plt.plot(ths_tot_2[0], ths_tot_2[1], 'bo-', label="important latter")
# # plt.ylim(0.0, 1.0)
# # plt.legend()
plt.savefig("result_sum.png")
plt.show()
