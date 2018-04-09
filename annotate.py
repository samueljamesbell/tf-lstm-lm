import numpy as np

import data


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        pairs = []
        for line in f:
            if line.strip():
                token, label = line.strip().split('\t')
            else:
                token, label = None, None

            pairs.append((token, label))

    return pairs


def pairs_to_lm_format(pairs):
    lines = ' '.join([t if t else '\n' for t, _ in pairs])
    tokens = data.preprocess_text(lines)
    return tokens


def write_file(path, pairs, rows, dimensions):
    empty = np.zeros(dimensions)
    num_rows = len(rows)

    with open(path, 'w') as f:
        for i, (t, c) in enumerate(pairs):
            if t.strip() and c.strip():
                row = rows[i] if i < num_rows else empty
                joined = '\t'.join(row.astype(str))
                f.write('{}\t{}\t{}\n'.format(t.strip(), joined, c.strip()))
            else:
                f.write()
