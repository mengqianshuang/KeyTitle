import rouge
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import BartTokenizerFast, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from data_load import BugDataset
import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def cut_word(sen_batch, word):
    res = []
    for sen in sen_batch:
        sen_copy = sen
        idx = sen_copy.find(word)
        if idx != -1:
            sen_copy = sen_copy[idx+len(word)+1 : ]
        res.append(sen_copy)
    return res

def gen_file(save_path1, save_path2, plm, tokenizer, device):
    valid_dataloader = DataLoader(
        BugDataset(data_dir=config.dataset_dir, dataset=config.test_dataset_name,
                   tokenizer=tokenizer),
        batch_size=16,
        num_workers=0,
        shuffle=True,
        drop_last=False,
        pin_memory=True)
    gen_text = []
    ref = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            text_input_idss, text_attention_masks, labelss, labels_attention_masks, summaries = \
            batch['text_input_ids'], \
            batch['text_attention_mask'], \
            batch['labels'], \
            batch['labels_attention_mask'], \
            batch['summary']

            text_input_idss = text_input_idss.to(device)
            text_attention_masks = text_attention_masks.to(device)
            generated_ids = plm.generate(input_ids=text_input_idss,
                                         attention_mask=text_attention_masks,
                                         max_length=config.max_seq_length,
                                         num_beams=config.num_beams,
                                         min_length=8
                                         )
            generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # generated = cut_word(generated, '[sentence]')
            # summaries = cut_word(summaries, '[sentence]')
            for idx, item in enumerate(generated):
                if len(item) == 0:
                    generated[idx] = "<unk>"
            gen_text.extend(generated)
            summaries = [x.replace("\n", "").replace("\r", "") for x in summaries]
            ref.extend(summaries)

    fout1 = open(save_path1, "w", encoding="utf8")
    fout2 = open(save_path2, "w", encoding="utf8")
    generated = cut_word(gen_text, '[sentence]')
    summaries = cut_word(ref, '[sentence]')
    for i in range(len(generated)):
        fout1.write(generated[i] + " \n")
        fout2.write(summaries[i] + " \n")
    fout1.close()
    fout2.close()


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t5_name_2 = "./model_save/bart-base_apache_sum_8"

    T5_tokenizer2 = BartTokenizerFast.from_pretrained(t5_name_2)
    T5_model2 = BartForConditionalGeneration.from_pretrained(t5_name_2)

    T5_model2.to(device)

    save_path2 = "./eval_data/res2.txt"
    ref_path = "./eval_data/ref.txt"

    gen_file(save_path2, ref_path, T5_model2, T5_tokenizer2, device)


    from rouge import FilesRouge
    files_rouge = FilesRouge()
    scores2 = files_rouge.get_scores(save_path2, ref_path, avg=True)
    print(scores2)


if __name__ == '__main__':
    main()