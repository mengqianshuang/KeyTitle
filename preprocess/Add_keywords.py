import pandas as pd
from GNN_models import KeywordsExtract, StackTraceExtract, ApacheInfoExtract
from tqdm.auto import tqdm
import spacy

en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words

df_train = pd.read_csv("../data/iTAPE_eclipse_train.csv")
df_val = pd.read_csv("../data/iTAPE_eclipse_valid.csv")
df_test = pd.read_csv("../data/iTAPE_eclipse_test.csv")
# df_train = pd.read_csv("../data/iTAPE_apache_train.csv")
# df_val = pd.read_csv("../data/iTAPE_apache_valid.csv")
# df_test = pd.read_csv("../data/iTAPE_apache_test.csv")
df_train = df_train.dropna()
df_val = df_val.dropna()
df_test = df_test.dropna()

# def replacecl(pd1: pd.DataFrame):
#     df_new = pd.DataFrame(data=None, columns=['summary', 'text'])
#     idx1 = 0
#     for idx, row in tqdm(pd1.iterrows()):
#         sum = row['summary'].replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
#         text = row['text'].replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
#
#         df_new.loc[idx1] = [sum, text]
#         idx1 += 1
#     return df_new



def add_keywords(file: pd.DataFrame):
    ke = KeywordsExtract()
    ste = StackTraceExtract()
    aie = ApacheInfoExtract()

    df_sum = pd.DataFrame(data=None, columns=['summary', 'text'])
    df_sum_gen = pd.DataFrame(data=None, columns=['summary', 'text'])
    file = file.replace('\t', ' ', regex=True).replace('\n', ' ', regex=True).replace('\r', ' ', regex=True)
    # data_row = self.data.iloc[idx]
    #
    # origin_text = data_row['text']
    # origin_sum = data_row['summary']
    #
    # text_keywords = self.getKeywords.get_keywords_by_order(origin_text)
    # sum_keywords = self.getKeywords.get_keywords_by_order(origin_sum)
    #
    # keyword_text = "[keywords] " + " ".join(text_keywords) + " [sentence] " + origin_text
    # keyword_sum = "[keywords] " + " ".join(sum_keywords) + " [sentence] " + origin_sum
    trk_num = 0
    stk_num = 0
    aie_num = 0
    idx_sum_gen = 0
    real_num = 0
    for idx, row in tqdm(file.iterrows()):
        sum = row['summary'].replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        text = row['text'].replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

        sum_keywords2 = ke.get_keywords(text)
        stack_trace_keywords2 = ste.extract(text)
        aie_keywords2 = aie.extract(text)


        sum_keywords = []
        stack_trace_keywords = []
        aie_keywords = []

        for token in sum_keywords2:
            if token.lower() not in stopwords:
                sum_keywords.append(token)
        for token in stack_trace_keywords2:
            if token.lower() not in stopwords:
                stack_trace_keywords.append(token)
        for token in aie_keywords2:
            if token.lower() not in stopwords:
                aie_keywords.append(token)

        keywords = set(sum_keywords + stack_trace_keywords + aie_keywords)
        keywords = [x.lower() for x in keywords]
        real_keywords = []
        if len(sum_keywords) > 0:
            trk_num += 1
        if len(stack_trace_keywords) > 0:
            stk_num += 1
        if len(aie_keywords) > 0:
            aie_num += 1
        for title_words in sum.split():
            if title_words.lower() in keywords:
                real_keywords.append(title_words)
        if len(real_keywords) > 0:
            real_num += 1
        keyword_sum = "[keywords] " + " ".join(real_keywords) + " [sentence] " + sum
        # keyword_text = "[keywords] " + " ".join(text_keywords) + " [sentence] " + text
        text = "summarize: " + text
        df_sum.loc[idx] = [keyword_sum, text]

        df_sum_gen.loc[idx_sum_gen] = [keyword_sum, text]
        idx_sum_gen += 1

        if len(real_keywords) > 0:
            source = "generate: " + "[keywords] " + " ".join(real_keywords)
            target = keyword_sum
            df_sum_gen.loc[idx_sum_gen] = [target, source]
            idx_sum_gen += 1

    print("文件总长度为: ", len(file))
    print("抽取到textRank keywords的文件数为: ", trk_num)
    print("抽取到stackTrace keyword的文件数为: ", stk_num)
    print("抽取到ApacheInfo keyword的文件数为: ", aie_num)
    print("抽取到real keyword的文件数为: ", real_num)
    return df_sum, df_sum_gen
# print(len(df_train))
# df_train = replacecl(df_train)
# print(len(df_train))
df_train_sum, df_train_sum_gen = add_keywords(df_train)
df_val_sum, df_val_sum_gen = add_keywords(df_val)
df_test_sum, df_test_sum_gen = add_keywords(df_test)

print("sum: train {}, val {}, test {} ".format(len(df_train_sum), len(df_val_sum), len(df_test_sum)))
print("sum_gen: train {}, val {}, test {} ".format(len(df_train_sum_gen), len(df_val_sum_gen), len(df_test_sum_gen)))
df_total_sum = pd.concat([df_train_sum, df_val_sum, df_test_sum], axis=0, ignore_index=True)
df_total_sum_gen = pd.concat([df_train_sum_gen, df_val_sum_gen, df_test_sum_gen], axis=0, ignore_index=True)
print("total: sum {}, sum_gen {} ".format(len(df_total_sum), len(df_total_sum_gen)))

df_total_sum.to_csv("../data/iTAPE_eclipse_total_sum.csv", index=False)
df_train_sum.to_csv("../data/iTAPE_eclipse_train_sum.csv", index=False)
df_val_sum.to_csv("../data/iTAPE_eclipse_val_sum.csv", index=False)
df_test_sum.to_csv("../data/iTAPE_eclipse_test_sum.csv", index=False)

df_total_sum_gen.to_csv("../data/iTAPE_eclipse_total_sum_gen.csv", index=False)
df_train_sum_gen.to_csv("../data/iTAPE_eclipse_train_sum_gen.csv", index=False)
df_val_sum_gen.to_csv("../data/iTAPE_eclipse_val_sum_gen.csv", index=False)
df_test_sum_gen.to_csv("../data/iTAPE_eclipse_test_sum_gen.csv", index=False)


