import os
import sys
import numpy as np
import random
import torch
from torch import nn
import torch.utils.data as Data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from configparser import ConfigParser
import warnings
import argparse

import esm
from antiberty import AntiBERTyRunner

sys.path.append("")
from utils import EarlyStopping, calculate_performance
from MyData import MyData
from Model import ESMberty_model

warnings.filterwarnings("ignore")


class ESMberty_Classifier():
    def __init__(self, config_path, Seed):
        self.config_path = config_path
        self.config = ConfigParser()
        self.config.read(config_path, encoding='UTF-8')
        self.seed = Seed

        self.input_data = self.config['path']['input_data']
        self.BATCH_SIZE = self.config.getint('parameter', 'batch')
        self.EPOCH = self.config.getint('parameter', 'epoch')
        self.patience = self.config.getint('parameter', 'patience')

        self.tag = self.config['output']['tag']
        self.log_dir = self.config['output']['log_dir']
        self.model_file = self.config['output']['model_dir']
        self.result_file = self.config['output']['result_dir']

        self.set_seed()

        self.board_tag_train = 'train_loss_batch{}_{}_seed{}_train_loss'.format(self.BATCH_SIZE, self.tag, self.seed)
        self.board_tag_val = 'val_loss_batch{}_{}_seed{}_valid_loss'.format(self.BATCH_SIZE, self.tag, self.seed)

        if os.path.exists(os.path.join(self.log_dir)) == False:
            os.makedirs(self.log_dir)
        if os.path.exists(os.path.join(self.model_file)) == False:
            os.makedirs(self.model_file)
        if os.path.exists(os.path.join(self.result_file)) == False:
            os.makedirs(self.result_file)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['other']['gpus']
        self.device = torch.device('cuda:0')
        torch.cuda.empty_cache()

        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.antiberty = AntiBERTyRunner()
        self.model.to(self.device)

    def pad_both(self, data_lst, max_len):
        out_lst = []
        for data in data_lst:
            encode_len = data.shape[0]
            add = max_len - encode_len
            pad = (0, 0, add // 2, add - (add // 2))
            out_lst.append(torch.nn.functional.pad(data, pad, 'constant', 0))
        return torch.stack(out_lst)

    def loader_dp(self, batch):
        label_lst, virus_lst, heavy_lst, light_lst = [], [], [], []
        for label, virus_seq, heavy_seq, light_seq in batch:
            virus_encode = [(str(label), virus_seq)]
            _, _, virus_tokens = self.batch_converter(virus_encode)
            with torch.no_grad():
                results_virus = self.model(virus_tokens.cuda(self.device), repr_layers=[33], return_contacts=True)

            virus_embedding = self.pad_both(results_virus["representations"][33], 256)
            heavy_embedding = self.pad_both(self.antiberty.embed([heavy_seq]), 256)
            light_embedding = self.pad_both(self.antiberty.embed([light_seq]), 256)
            label_lst.append(torch.tensor(label))
            virus_lst.append(virus_embedding)
            heavy_lst.append(heavy_embedding)
            light_lst.append(light_embedding)
        return torch.stack(label_lst, dim=0), torch.stack(virus_lst, dim=0), torch.stack(heavy_lst, dim=0), torch.stack(
            light_lst, dim=0)

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
    
    def train(self):
        print("start train")
        checkpoint = os.path.join(self.model_file,
                                  'batch{}_{}_checkpoint.pt'.format(self.BATCH_SIZE, self.tag))
        result_file = os.path.join(self.result_file,
                                   'batch{}_{}.csv'.format(self.BATCH_SIZE, self.tag))
        writer = SummaryWriter(log_dir=self.log_dir)

        train_data = MyData(self.input_data, split='train')
        valid_data = MyData(self.input_data, split='eval')

        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, collate_fn=self.loader_dp,                               
                                       shuffle=True)
        valid_loader = Data.DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, collate_fn=self.loader_dp,
                                       shuffle=True)

        ESMberty = ESMberty_model()
        ESMberty.to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(ESMberty.parameters(), lr=0.001)
        early_stopping = EarlyStopping(self.patience, verbose=True, path=checkpoint)

        train_step = 0
        val_step = 0
        for epoch in range(self.EPOCH):
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for step, (label, virus, heavy, light) in loop:
                ESMberty.train()
                output = ESMberty(virus.to(self.device), heavy.to(self.device), 
                                                       light.to(self.device))
                output = output.view(-1).to(torch.float)
                loss = criterion(output, label.to(torch.float).to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_step += 1
                writer.add_scalar(tag=self.board_tag_train, scalar_value=loss, global_step=train_step)
                loop.set_description(f'Epoch [{epoch}/{self.EPOCH}]')

            all_pred = []
            all_val = []
            loop_val = tqdm(enumerate(valid_loader), total=len(valid_loader))
            for step, (label, virus, heavy, light) in loop_val:
                ESMberty.eval()
                output = ESMberty(virus.to(self.device), heavy.to(self.device),
                                                       light.to(self.device))
                output = output.view(-1).to(torch.float)
                loss = criterion(output, label.to(torch.float).to(self.device))
                val_step += 1
                writer.add_scalar(tag=self.board_tag_val, scalar_value=loss, global_step=val_step)
                loop_val.set_description(f'Epoch [{epoch}/{self.EPOCH}]')
                all_pred.extend(output.cpu().detach().numpy().tolist())
                all_val.extend(label.cpu().detach().numpy().tolist())

            early_stopping(loss, ESMberty)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            acc, pos_acc, neg_acc = calculate_performance(np.array(all_val), np.array(all_pred), epoch,
                                                          out_file=result_file)
            print('Epoch: ', epoch, '| val loss: %.8f' % loss, '| val acc: %.2f' % acc, '| val pos_acc: %.2f' % pos_acc,
                  '| val neg_acc: %.2f' % neg_acc)
        writer.close()

    def test(self, type, subtype, checkpoint=None):
            print("start test")
            if checkpoint is None:
                checkpoint = os.path.join(self.model_file,
                                        'batch{}_{}_checkpoint.pt'.format(self.BATCH_SIZE, self.tag))
            result_file = os.path.join(self.result_file,
                    'test_batch{}_{}.csv'.format(self.BATCH_SIZE, self.tag))
            test_data = MyData(self.input_data, split=type, type=subtype)

            test_loader = Data.DataLoader(dataset=test_data, batch_size=self.BATCH_SIZE, collate_fn=self.loader_dp,
                                        shuffle=False)
            ESMberty = ESMberty_model()
            ESMberty.load_state_dict(torch.load(checkpoint), strict=False)
            ESMberty.to(self.device)

            all_pred = []
            all_test = []
            loop_test = tqdm(enumerate(test_loader), total=len(test_loader))
            for step, (label, virus, heavy, light) in loop_test:
                ESMberty.eval()
                output = ESMberty(virus.to(self.device), 
                                heavy.to(self.device),
                                light.to(self.device))
                output = output.view(-1).to(torch.float)

                all_pred.extend(output.cpu().detach().numpy().tolist())
                all_test.extend(label.cpu().detach().numpy().tolist())

            acc, pos_acc, neg_acc = calculate_performance(np.array(all_test), np.array(all_pred), type + subtype,
                                                        out_file=result_file)
            print('Epoch: ', type + subtype, '| test acc: %.2f' % acc, '| test pos_acc: %.2f' % pos_acc,
                '| test neg_acc: %.2f' % neg_acc)

            if os.path.exists(os.path.join(self.model_file, type, subtype)) == False:
                os.makedirs(os.path.join(self.model_file, type, subtype))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''PLMABFW is a cutting-edge predictive framework designed to analyze antigen-antibody interactions. 
        It leverages protein language model encoding to provide accurate predictions. 
        To use PLMABFW, simply specify the path to your config file and the desired operation, and the framework will execute accordingly.''',
        epilog='''For more details and guidance on how to configure and run PLMABFW, 
        please refer to our documentation on GitHub:https://github.com/Chenyb939/PLMABFW'''
    )

    parser.add_argument('mode', help='Mode of operation, either "train" or "test".', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--config', help='Path to the config file. default ./config', type=str, default='./config')
    parser.add_argument('--seed', help='Random seed. default 42', type=int, default=42)

    args = parser.parse_args()

    ESMberty = ESMberty_Classifier(args.config, Seed=args.seed, mode=args.mode)
    if args.mode == 'train':
        ESMberty.train()
    elif args.mode == 'test':
        ESMberty.test()
