from . import vocabulary


def print_vocab():
    header = 'Word | Default Selector | Parameters | Description\n' \
                + ':---|:---|:---|:---\n'
    this_cat = ''
    cat_order = {}
    i = 0
    for cat,_,_ in main.vocab.values():
        if cat not in cat_order:
            cat_order[cat] = str(i).zfill(2)
            i += 1
    for name, (cat, desc, factory) in sorted(main.vocab.items(), \
            key=lambda i: cat_order[i[1][0]] + i[0]):
        if cat != this_cat:
            print('### ' + cat + '\n')
            print(header)
            this_cat = cat
        word = factory(name)
        s = word.slice_
        params = ', '.join([f'_{c}_' for c in word.__call__.__code__.co_varnames \
                if c not in ['self','kwargs','key','value']])
        sel = f'[{s.start if s.start else ''}:{s.stop if s.stop else ''}]'
        print(f'{name}|{sel}|{params}|{desc}\n')

if __name__ == '__main__':
    print_vocab()
