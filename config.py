import torch
seed = 19991216
reproducibility = True
dataset_dir = "./data/"

train_dataset_keywords_name = "eclipse_train_sum.csv"
valid_dataset_keywords_name = "eclipse_val_sum.csv"
test_dataset_keywords_name = "eclipse_test_sum.csv"
test_dataset_keywords_name_gen = "eclipse_test_sum_gen.csv"

train_dataset_name = "iTAPE_eclipse_train.csv"
valid_dataset_name = "iTAPE_eclipse_valid.csv"
test_dataset_name_sum = "iTAPE_eclipse_test.csv"
test_dataset_name_gen = "eclipse_test_gen.csv"

epochs = 10

model_dir_t5 = "./Model/t5-base"
model_dir_bart_base = "./Model/bart-base"
model_dir_gpt2 = "./Model/gpt2"

text_col_name = "description"
des_col_name = "summary"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_lr = 1e-5
batch_size = 1

num_beams = 12
max_seq_length = 100

loss_name = "loss"
loss_name_val = "loss_val"
lr_p = "lr"
save_model_name = "gpt2"

