import random
random.seed(42)
DATA_SIZE = 19186
TRAIN_VAL_TEST = [0.9, 0, 0.1]


def split(n_total, train_ratio, var_ratio):
    lines = list(range(0, n_total))

    train_offset = int(n_total * train_ratio)
    val_offset = int(n_total * (train_ratio + var_ratio)) + 50
    random.shuffle(lines)
    prefix_path = './mtorl/datasets/occ_split/' + str(n_total) + '_'
    train_data = open(prefix_path+'occ_9p.txt', 'w+')
    val_data = open(prefix_path+'occ_50.txt', 'w+')
    test_data = open(prefix_path+'occ_1p.txt', 'w+')

    for i, line in enumerate(lines):
        line = str(line).zfill(5)
        if i < train_offset:
            train_data.write(line)
            train_data.write("\n")
        elif i < val_offset:
            val_data.write(line)
            val_data.write("\n")
        else:
            test_data.write(line)
            test_data.write("\n")
    train_data.close()
    val_data.close()
    test_data.close()


if __name__ == "__main__":
   split(DATA_SIZE, TRAIN_VAL_TEST[0], TRAIN_VAL_TEST[1])

