def get_all_folds_names():
    folds = []
    for i in range(1, 11):
        f = open(f'./data/experimental_protocols/fold_all/fold_all_{i}.txt', 'r')
        lines = f.readlines()
        f.close()
        folds.append(list(map(str.strip, lines[2:])))
    return folds
