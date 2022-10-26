import numpy as np
import itertools
import sys
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead
from datetime import datetime
from easydict import EasyDict
import os
import torch
from transformers import GPT2LMHeadModel, AutoModelWithLMHead
from transformers import GPT2Tokenizer, PreTrainedTokenizerFast, AutoTokenizer
import pymysql

con = pymysql.connect(host='diary.cu6lgsxzmymo.ap-northeast-2.rds.amazonaws.com',
                      port=3306,
                      user='diary',
                      password='rnfma^123',
                      db='diary',
                      charset='utf8')

cur = con.cursor()


tokenizer = AutoTokenizer.from_pretrained("snoop2head/KoGPT-Joong-2")
model = AutoModelWithLMHead.from_pretrained("snoop2head/KoGPT-Joong-2")

# root path
ROOT_PATH = os.path.abspath(".") # this makes compatible absolute path both for local and server

# designate root path for the data
DATA_ROOT_PATH = os.path.join(ROOT_PATH, 'data')

# designate path for each dataset files
LYRIC_PATH = os.path.join(DATA_ROOT_PATH, "lyrics_kor.txt")
BILLBOARD_PATH = os.path.join(DATA_ROOT_PATH, "rawdata_김지훈_201500844.tsv")
GEULSTAGRAM_PATH = os.path.join(DATA_ROOT_PATH, "geulstagram.csv")

# Initialize configuration
CFG = EasyDict()

# Dataset Config as constants
CFG.DEBUG = False
CFG.num_workers = 4
CFG.train_batch_size = 16

# Train configuration
CFG.user_name = "snoop2head"
today = datetime.now().strftime("%m%d_%H:%M")
CFG.file_base_name = f"{CFG.user_name}_{today}"
CFG.model_dir = "skt/ko-gpt-trinity-1.2B-v0.5" # designate the model's name registered on huggingface: https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5
CFG.max_token_length = 42
CFG.learning_rate = 5e-5
CFG.weight_decay = 1e-2 # https://paperswithcode.com/method/weight-decay

# training steps configurations
CFG.save_steps = 500
CFG.early_stopping_patience = 5
CFG.warmup_steps = 500
CFG.logging_steps = 100
CFG.evaluation_strategy = 'epoch'
CFG.evaluation_steps = 500

# Directory configuration
CFG.result_dir = os.path.join(ROOT_PATH, "results")
CFG.saved_model_dir = os.path.join(ROOT_PATH, "best_models")
CFG.logging_dir = os.path.join(ROOT_PATH, "logs")
CFG.baseline_dir = os.path.join(ROOT_PATH, 'baseline-code')

# read txt file from line by line
def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

# make sampling function from the list
def sampling(list_lines:list, n:int) -> list:
    # sampling
    list_lines = np.random.choice(list_lines, n)
    list_lines = list(list_lines)
    return list_lines

# CFG.saved_model_dir = "./results"
CFG.model_dir = "snoop2head/KoGPT-Joong-2"

# Attach Language model Head to the pretrained GPT model
model = AutoModelWithLMHead.from_pretrained(CFG.model_dir) # KoGPT3 shares the same structure as KoGPT2.


# move the model to device
if torch.cuda.is_available() and CFG.DEBUG == False:
    device = torch.device("cuda:0")
elif CFG.DEBUG == True or not torch.cuda.is_available():
    device = torch.device("cpu")

model.to(device)
model.eval()

# https://huggingface.co/transformers/preprocessing.html
# Load the Tokenizer: "Fast" means that the tokenizer code is written in Rust Lang
tokenizer = AutoTokenizer.from_pretrained(
    CFG.model_dir,
    max_len = CFG.max_token_length,
    padding='max_length',
    add_special_tokens = True,
    return_tensors="pt",
    truncation = True,
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    pad_token = "<pad>",
    mask_token = "<mask>",
)


def infer_sentence(input_sentence, k, output_token_length):
    # encode the sample sentence
    input_ids = tokenizer.encode(
        input_sentence,
        add_special_tokens=False,
        return_tensors="pt"
    )

    # decode the output sequence and print its outcome
    list_decoded_sequences = []
    while len(list_decoded_sequences) < k:
        # generate output sequence from the given encoded input sequence
        output_sequences = model.generate(
            input_ids=input_ids.to(device),
            do_sample=True,
            max_length=output_token_length,
            num_return_sequences=k
        )

        for index, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            # remove padding from the generated sequence
            generated_sequence = generated_sequence[:generated_sequence.index(tokenizer.pad_token_id)]
            decoded_sequence = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # print(f"{index} : {decoded_sequence}")
            list_decoded_sequences.append(decoded_sequence)
        list_decoded_sequences = list(set(list_decoded_sequences))

    return list_decoded_sequences


def make_residual_samhaengshi(input_letter, k, output_token_length):
    # make letter string into
    list_samhaengshi = []

    # initializing text and index for iteration purpose
    index = 0

    # iterating over the input letter string
    for index, letter_item in enumerate(input_letter):
        # initializing the input_letter
        if index == 0:
            residual_text = letter_item
        else:
            pass

        # infer and add to the output
        list_sentences = infer_sentence(residual_text, 3, output_token_length)
        for sentence in list_sentences:
            if len(sentence) == 1:
                pass
            elif len(sentence) >= 2:
                inferred_sentence = sentence  # first item of the inferred list
        if index != 0:
            # remove previous sentence from the output
            inferred_sentence = inferred_sentence.replace(list_samhaengshi[index - 1], "").strip()
        else:
            pass
        list_samhaengshi.append(inferred_sentence)

        # until the end of the input_letter, give the previous residual_text to the next iteration
        if index < len(input_letter) - 1:
            residual_sentence = list_samhaengshi[index]
            next_letter = input_letter[index + 1]
            residual_text = f"{residual_sentence} {next_letter}"  # previous sentence + next letter
            # print(residual_text)

        elif index == len(input_letter) - 1:  # end of the input_letter
            # Concatenate strings in the list without intersection

            return list_samhaengshi

def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings,
                                             candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def main(argv):
    print(argv[1] + " + " + argv[2])
    f = open(argv[1], 'r', encoding='UTF8')
    lines = f.readlines()
    f.close

    user1 = ""
    user2 = ""
    text = []

    for line in lines[5:]:
        line = line.replace('\n', '')
        line = line.split(" : ")
        if len(line) == 1:
            pass
        else:
            text.append(line[1])
            user = line[0].split()[-1]
            user = user.strip()
            if user != "":
                if user1 == "":
                    user1 = user
                elif user2 == "" and user1 != user:
                    user2 = user

    lines = ' '.join(text)

    okt = Okt()

    tokenized_doc = okt.pos(lines)
    tokenized_nouns = [word[0] for word in tokenized_doc if word[1] == 'Noun']

    result_list = []

    stop_words = "".split(' ')

    for w in tokenized_nouns:
        if w not in stop_words:
            if len(w) != 1:
                result_list.append(w)

    tokenized_nouns = ' '.join(result_list)


    count = CountVectorizer(ngram_range=(1, 1)).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    doc_embedding = model.encode([lines])
    candidate_embeddings = model.encode(candidates)

    top_n = 8
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    global word
    word = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=3, nr_candidates=30)
    for w in word:
        print(w)
    inferred_samhaengshi = make_residual_samhaengshi(word, k=1, output_token_length=CFG.max_token_length)
    sentence = ""
    for item in inferred_samhaengshi:
        sentence = sentence + item + " "
    print(sentence)
    query = "update diary set content = %s, status = 2 where diary_id = %s"
    cur.execute(query, (sentence, argv[2]))
    con.commit()
    print(query + sentence + argv[2])
    con.close()



if __name__ == '__main__':
    main(sys.argv)

