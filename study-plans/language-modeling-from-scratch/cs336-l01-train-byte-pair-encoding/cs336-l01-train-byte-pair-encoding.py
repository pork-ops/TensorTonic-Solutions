def find_merge(d: dict, merges: list, vocab: list, base_vocab: int):
    max_ocurrences = 0
    max_key = None
    for key in d:
        if d[key] > max_ocurrences:
            max_ocurrences = d[key]
            max_key = key
        elif d[key] == max_ocurrences:
            key_bytes = (get_original_ids(vocab, key[0], base_vocab), get_original_ids(vocab, key[1], base_vocab))
            max_key_bytes = (get_original_ids(vocab, max_key[0], base_vocab), get_original_ids(vocab, max_key[1], base_vocab))
            if max_key_bytes < key_bytes:
                max_ocurrences = d[key]
                max_key = key

    return max_key


def replace_merges(corpus, letter_1, letter_2, new_id):
    new_corpus = []
    for word in corpus:
        jump = False
        new_word = []
        for i in range(len(word) - 1):
            # print(i, word[i])
            if jump: 
                jump = False
                continue
            if letter_1 == word[i] and letter_2 == word[i+1]:
                new_word.append(new_id)
                jump = True
            else: 
                new_word.append(word[i])
        if not jump and word: # prevent empty word
            new_word.append(word[-1])
        new_corpus.append(new_word)
    corpus = new_corpus
    return corpus


def calculate_ocurrences(corpus):
    d = dict()
    for word in corpus:
        for (l1, l2) in zip(word[0::], word[1::]):
            if (l1, l2) not in d:
                d[(l1, l2)] = 1
            else: d[(l1, l2)] = d[(l1, l2)] + 1        
    return d


def get_original_ids(vocabulary: list[str], token_id: int, base_vocab: int):
    if token_id >= base_vocab:
        for t in vocabulary:
            if t[0] == token_id:
                return t[1]
    return [token_id]


def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    base_vocab = 256
    vocab = []
    merges = []

    new_corpus = []
    for word in corpus:
        new_corpus.append(word.encode('utf-8'))
    corpus = new_corpus
    
    while base_vocab + len(vocab) < vocab_size:
        print("Looking for more...", base_vocab + len(vocab))

        # Looking for another merge...
        d = calculate_ocurrences(corpus=corpus)
        #print(d)

        # Merge
        next_merge = find_merge(d, merges, vocab, base_vocab)
        if next_merge is None:
            break
        current_vocab = base_vocab + len(vocab)
        merges.append([next_merge[0], next_merge[1], current_vocab])
        vocab.append([current_vocab, get_original_ids(vocab, next_merge[0], base_vocab) + get_original_ids(vocab, next_merge[1], base_vocab)])

        # Replace merged letters
        corpus = replace_merges(corpus, next_merge[0], next_merge[1], current_vocab)

        print({"vocab": vocab, "merges": merges})
    return {"vocab": vocab, "merges": merges}
