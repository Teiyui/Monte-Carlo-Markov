import random

#
dice_probs_list = [0.1, 0.15, 0.25, 0.35, 0.05, 0.1]
def sample(probs_list, iterations = 10000):
    list = probs_list
    kv = {}

    for m in range(len(list)):
        kv[str(m)] = 0

    for i in range(1, len(list)):
        list[i] += list[i-1]

    for j in range(iterations):
        random_val = random.uniform(0, 1)*list[-1]
        for t in range(len(list)):
            if random_val <= list[t]:
                kv[str(t)] += 1
                break

    return kv

kv = sample(dice_probs_list)
print(kv)


