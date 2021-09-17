# coding:UTF-8
# author    :Just_silent
# init time :2021/5/24 17:42
# file      :path.py
# IDE       :PyCharm

system = 'email'

xlsx_path = r'D:\bf_workspace\data\{}\{}.xlsx'.format(system, system)
train_data_path = r'D:\bf_workspace\data\{}\data.txt'.format(system)

# pre_train_model
vocab_path = r'D:\bf_workspace\pre_trained_model\chinese_L-12_H-768_A-12\vocab.txt'
config_path = r'D:\bf_workspace\pre_trained_model\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\bf_workspace\pre_trained_model\chinese_L-12_H-768_A-12\bert_model.ckpt'
bert_base_model_save_path = r'D:\bf_workspace\saved_model\{}\bbm.weights'.format(system)
simCSE_model_save_path = r'D:\bf_workspace\saved_model\{}\simCSE.weights'.format(system)
train_path = r'D:\bf_workspace\data\simi_new\simi_new.train.data'
valid_path = r'D:\bf_workspace\data\simi_new\simi_new.valid.data'
test_path = r'D:\bf_workspace\data\simi_new\simi_new.test.data'