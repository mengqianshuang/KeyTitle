import config
from data_load import BugDataset, BugDatasetKeywords, BugDatasetKeywords2
import os
import torch
import time
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch import optim
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def init_seed(seed, reproducibility):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def run_train_batch(batch, plm,
                    plm_optimizer, scheduler, scaler, device):

    text_input_idss, text_attention_masks, labelss, labels_attention_masks = batch['text_input_ids'], batch['text_attention_mask'], batch['labels'], batch['labels_attention_mask']

    if torch.cuda.is_available():
        text_input_idss = text_input_idss.to(device)
        text_attention_masks = text_attention_masks.to(device)

        labelss = labelss.to(device)
        labels_attention_masks = labels_attention_masks.to(device)

    # with torch.cuda.amp.autocast():
    output_dict = plm(input_ids=text_input_idss,
                      attention_mask=text_attention_masks,
                      decoder_attention_mask=labels_attention_masks,
                      labels = labelss,
                      return_dict=True)

    gen_loss = output_dict["loss"]

    plm_optimizer.zero_grad()

    scaler.scale(gen_loss).backward()
    scaler.step(optimizer=plm_optimizer)
    scheduler.step()
    scaler.update()


    # gen_loss.backward()
    # plm_optimizer.step()
    # scheduler.step()


    return gen_loss.item()


def run_eval_batch(batch, plm, plm_optimizer, scheduler, device):
    text_input_idss, text_attention_masks, labelss, labels_attention_masks = batch['text_input_ids'], batch['text_attention_mask'], batch['labels'], batch['labels_attention_mask']
    if torch.cuda.is_available():
        text_input_idss = text_input_idss.to(device)
        text_attention_masks = text_attention_masks.to(device)

        labelss = labelss.to(device)
        labels_attention_masks = labels_attention_masks.to(device)

    output_dict = plm(input_ids=text_input_idss,
                      attention_mask=text_attention_masks,
                      decoder_attention_mask=labels_attention_masks,
                      labels=labelss,
                      return_dict=True)

    gen_loss = output_dict["loss"]
    return gen_loss.item()


def train(writer):

    print("Start!")
    # init_seed(config["seed"], config["reproducibility"])
    device = config.device

    print("Build PLM Model.")
    T5_tokenizer =  T5TokenizerFast.from_pretrained(config.model_dir_t5)
    plm = T5ForConditionalGeneration.from_pretrained(config.model_dir_t5)
    plm.to(device)

    T5_tokenizer.add_tokens(['[keywords]', '[sentence]'])
    plm.resize_token_embeddings(len(T5_tokenizer))

    plm_parameters = [p for p in plm.parameters() if p.requires_grad]
    plm_optimizer = optim.Adam(plm_parameters, config.train_lr)

    print("Create training dataset.")

    train_dataloader = DataLoader(
        BugDataset(data_dir=config.dataset_dir, dataset=config.train_dataset_name,
                   tokenizer=T5_tokenizer),
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        pin_memory=True)

    print("Create validation dataset.")
    valid_dataloader = DataLoader(
        BugDataset(data_dir=config.dataset_dir, dataset=config.valid_dataset_name,
                   tokenizer=T5_tokenizer),
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        pin_memory=True)

    best_gen_loss = None
    scaler_idx = 0
    scaler_idx_val = 0
    scaler = GradScaler()
    total_steps = len(train_dataloader)*config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=plm_optimizer,
                                                num_warmup_steps=int(0.1*total_steps),
                                                num_training_steps=total_steps)
    for epoch_idx in tqdm(range(config.epochs)):

        plm.train()


        train_gen_loss = 0
        t0 = time.time()
        for batch in tqdm(train_dataloader):
            gen_loss = run_train_batch(batch, plm, plm_optimizer, scheduler, scaler, device)
            # print("Epoch {} batch {}: Gen loss {}.".format(epoch_idx, batch_idx, gen_loss))

            train_gen_loss += gen_loss

            writer.add_scalar(config.loss_name, gen_loss, scaler_idx)
            writer.add_scalar(config.lr_p, plm_optimizer.param_groups[0]['lr'], scaler_idx)
            scaler_idx += 1

        train_gen_loss /= len(train_dataloader)
        train_ppl = np.exp(train_gen_loss)
        training_time = format_time(time.time() - t0)
        print("\n\nEpoch {}: training generation loss {} perplexity {} time {}.".format(epoch_idx,
                                                                                                train_gen_loss,
                                                                                                train_ppl,
                                                                                                training_time))

        with torch.no_grad():
            plm.eval()
            valid_gen_loss = 0
            t0 = time.time()
            for batch in tqdm(valid_dataloader):
                gen_loss = run_eval_batch(batch, plm, plm_optimizer, scheduler, device)
                writer.add_scalar(config.loss_name_val, gen_loss, scaler_idx_val)
                scaler_idx_val += 1
                valid_gen_loss += gen_loss

            valid_gen_loss /= len(valid_dataloader)
            valid_ppl = np.exp(valid_gen_loss)
            valid_time = format_time(time.time() - t0)
            print("\n\nEpoch {}: validation generation loss {} perplexity {} time {}.".format(epoch_idx,
                                                                                                      valid_gen_loss,
                                                                                                      valid_ppl,
                                                                                                      valid_time))

        if best_gen_loss is None or valid_gen_loss <= best_gen_loss:
            output_dir = '{}_sum_{}'.format(config.save_model_name, str(epoch_idx))
            saved_path = os.path.join("./model_save", output_dir)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            # save pretrained language model
            model_to_save = plm.module if hasattr(plm, 'module') else plm
            model_to_save.save_pretrained(saved_path)
            T5_tokenizer.save_pretrained(saved_path)

            best_gen_loss = valid_gen_loss

def main():
    writer = SummaryWriter('./logs')
    init_seed(config.seed, config.reproducibility)
    train(writer)
    writer.close()



if __name__ == '__main__':
    main()
