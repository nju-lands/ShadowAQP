import math
import os
import time
import numpy as np
import pandas as pd
import threading
import torch
from keras.utils.np_utils import to_categorical
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.utils.data import DataLoader
from torch import optim
from utils.model_utils import save_torch_model, load_torch_model
import logging
from utils.dataset_utils import *
from utils.pytorchtools import EarlyStopping

logger = logging.getLogger(__name__)


class CVAE(nn.Module):
    def __init__(self, data_dim, label_dim, intermediate_dim, latent_dim, dataset):
        super(CVAE, self).__init__()
        self.device = dataset.device
        self.numeric_flag = len(dataset.numeric_columns) > 0
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        # self.fc1 = nn.Linear(origin_dim, intermediate_dim)
        encoder_dim = [intermediate_dim]
        decoder_dim = [intermediate_dim]
        # encoder_dim = [intermediate_dim, intermediate_dim]
        # decoder_dim = [intermediate_dim, intermediate_dim]
        seq = []
        dim = data_dim + label_dim
        for item in encoder_dim:
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.encode_seq = Sequential(*seq)
        self.fc21 = nn.Linear(intermediate_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(intermediate_dim, latent_dim)  # var
        seq = []
        dim = latent_dim + label_dim
        for item in decoder_dim:
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.decode_seq = Sequential(*seq)
        self.fc4 = nn.Linear(intermediate_dim, data_dim)
        self.output_info = dataset.encoded_output_info

        self.sigmoid = nn.Sigmoid()

        # self.output_layers = nn.ModuleList(
        #     [nn.Linear(intermediate_dim, digit) for digit, activ in dataset.encoded_output_info])
        # self.output_activ = [info[1] for info in dataset.encoded_output_info]

        # self.sigma = []
        # if self.numeric_flag:
        #     self.sigma = nn.Parameter(torch.ones(data_dim + label_dim) * 0.1)
        # else:
        #     self.sigma = []

    def encode(self, x, c):
        # h1 = F.relu(self.fc1(x))
        inputs = torch.cat([x, c], 1)
        h1 = self.encode_seq(inputs)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, c):
        # h3 = F.relu(self.fc3(z))
        inputs = torch.cat([z, c], 1)
        h3 = self.decode_seq(inputs)
        output = self.fc4(h3)

        # sigma=self.sigma
        # if self.numeric_flag:
        #     sigma = self.sigma.clamp(-3, 3)  # p_params['logvar_x'] = self.logvar_x

        # sigma = Parameter(torch.ones(self.origin_dim) * 0.1)
        return output

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        return self.decode(z, c), mu, logvar  # 解码，同时输出均值方差

    def loss_function(self, recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """

        batch_size = x.size(0)
        numeric_loss = torch.tensor(0.0).to(self.device)
        categorical_loss = torch.tensor(0.0).to(self.device)
        st = 0
        for digit, activ in self.output_info:
            ed = st + digit
            if activ == 'sigmoid':
                # categorical_loss += F.binary_cross_entropy(recon_x[:, st:ed], x[:, st:ed], reduction='mean')
                categorical_loss += F.binary_cross_entropy(self.sigmoid(recon_x[:, st:ed]), x[:, st:ed], reduction='mean')
            elif activ == 'softmax':
                # categorical_loss += F.binary_cross_entropy(recon_x[:, st:ed], x[:, st:ed], reduction='mean')
                # categorical_loss += F.nll_loss(torch.log(recon_x[:, st:ed]), torch.argmax(x[:, st:ed], dim=-1),
                #                                reduction='mean')
                categorical_loss += F.cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1),
                                                    reduction='mean')
            else:
                numeric_loss += F.mse_loss(recon_x[:, st:ed],x[:, st:ed], reduction='mean')
                # numeric_loss += F.mse_loss(recon_x[:, st:ed], torch.tanh(x[:, st:ed]), reduction='mean')
            st = ed
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return numeric_loss + categorical_loss, kld, numeric_loss, categorical_loss


def alpha_schedule(epoch, max_epoch, alpha_max, strategy="exp"):
    # strategy to adjust weight
    if strategy == "linear":
        alpha = alpha_max * min(1, epoch / max_epoch)
    elif strategy == "exp":
        alpha = alpha_max * math.exp(-5 * (1 - min(1, epoch / max_epoch)) ** 2)
    else:
        raise NotImplementedError("Strategy {} not implemented".format(strategy))
    return alpha

def torch_cvae_train(model, dataset, learning_rate, epochs, batch_size):
    start_time = time.perf_counter()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    latent_param = {}
    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        train_loss_vae, train_kld_vae, train_num_vae, train_cat_vae = 4 * [0]
        alpha = alpha_schedule(epoch, 100, 0.8)
        # alpha = alpha_schedule(epoch, 50, 1)
        # alpha=1.0
        for batch_idx, input_data in enumerate(loader):
            optimizer.zero_grad()
            x, c = input_data
            recon_x, mu, logvar = model(x, c)
            recon_loss, kld, numeric_loss, categorical_loss = model.loss_function(recon_x, x, mu, logvar)
            loss = recon_loss + kld * alpha
            loss.backward()
            optimizer.step()
            # model.sigma.data.clamp_(0.01, 1.0)
            train_loss_vae += loss.item()
            train_kld_vae += kld.item()
            train_num_vae += numeric_loss.item()
            train_cat_vae += categorical_loss.item()

        epoch_end_time = time.perf_counter()
        logger.info('----------------------------No.{} epoch----------------------------'.format(epoch + 1))
        logger.info(
            'loss:{}, numeric_loss:{}, categorical_loss:{}, kld_loss:{}, epoch_train_time:{}'.format(
                train_loss_vae,
                train_num_vae,
                train_cat_vae,
                train_kld_vae,
                (epoch_end_time - epoch_start_time)))
        # early_stopping(loss, model)
        # # 若满足 early stopping 要求
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     # 结束模型训练
        #     break

    end_time = time.perf_counter()
    logger.info('training time elapsed:{}'.format(end_time - start_time))
    return model


def generate_samples(model, dataset, query_config, train_config):
    sample_rate = train_config["sample_rate"]
    if train_config['sample_method'] == "senate":
        sample_allocation, sample_rates = senate_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "house":
        sample_allocation, sample_rates = house_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "advance sencate":
        sample_allocation, sample_rates = advance_senate_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "statistics":
        # sample_allocation, sample_rates = statistics_sampling(model, dataset, sample_rate, query_config)
        sample_allocation, sample_rates = statistics_sampling_with_small_group(model, dataset, sample_rate, query_config)
        # print("sample_allocation before: ", sample_allocation)  
    elif train_config['sample_method'] == "statistics multi label":
        sample_allocation, sample_rates = statistics_sampling_with_multi_label(model, dataset, sample_rate, query_config)   
    elif train_config['sample_method'] == "statistics model optimize":
        sample_allocation, sample_rates = statistics_sampling_with_model_optimize(model, dataset, sample_rate, query_config, train_config)                                             
    else:
        sample_allocation, sample_rates = statistics_sampling_with_small_group(model, dataset, sample_rate,
                                                                               query_config)
    # print("sample_allocation: ",sample_allocation)

    samples = generate_samples_with_allocation(dataset, model, sample_allocation, sample_rates, train_config)
    # print("=====sample_allocation after: ", sample_allocation)
    # print("=====sample_rates: ", sample_rates)
    # print(samples[:10])

    # save samples, but bring I/O cost
    # samples.to_csv('./samples/'+train_config['name']+".csv",index=False)
    if 'outliers' in train_config and train_config['outliers'] == 'true':
        samples = pd.concat([samples, dataset.outliers])
        # samples.to_csv('./samples/'+train_config['name']+"_with_outlier.csv",index=False)
    # save_samples(samples, train_config)
    # samples=read_samples(train_config)
    return samples

def train_torch_cvae(train_config):
    # hyper parameters
    start_time = time.perf_counter()
    lr = train_config["lr"]
    optimizer_type = train_config["optimizer_type"]
    batch_size = train_config["batch_size"]
    latent_dim = train_config["latent_dim"]
    intermediate_dim = train_config["intermediate_dim"]
    epochs = train_config["epochs"]
    logger.info("epoch:{}".format(epochs))
    logger.info("batch size:{}".format(batch_size))
    logger.info("latent dimension:{}".format(latent_dim))
    logger.info("intermediate dimension:{}".format(intermediate_dim))
    logger.info("gpu num:{}".format(train_config['gpu_num']))
    if exist_dataset(train_config):
        dataset = load_dataset(train_config)
    else:
        dataset = TabularDataset(train_config)
    logger.info("feature info:{}".format(dataset.feature_info))
    _, data_dim = dataset.data.shape
    model = CVAE(data_dim, dataset.label_size, intermediate_dim, latent_dim, dataset)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(dataset.device)
    model = torch_cvae_train(model, dataset, learning_rate=lr, epochs=epochs, batch_size=batch_size)
    save_torch_model(model, train_config)
    save_dataset(dataset, train_config)
    end_time = time.perf_counter()
    logger.info("train model time elapsed:{}".format(end_time - start_time))
    return model, dataset

def load_model_and_dataset(train_config):
    start_time = time.perf_counter()
    postfix = ''
    if 'inc_train_flag' in train_config and train_config['inc_train_flag'] != 'origin_train':
        postfix = train_config['inc_train_flag']
    dataset = load_light_dataset(train_config, postfix=postfix)
    if dataset is None:
        logger.error("dataset file not found")
        return None,None
    logger.info("feature info:{}".format(dataset.feature_info))
    latent_dim = train_config["latent_dim"]
    intermediate_dim = train_config["intermediate_dim"]
    model = CVAE(dataset.numeric_digits + dataset.categorical_digits, dataset.label_size, intermediate_dim, latent_dim,
                 dataset)
    model = load_torch_model(train_config, model, postfix=postfix)
    # model.to(dataset.device)
    if model is None:
        logger.error("model file not found")
    end_time = time.perf_counter()
    logger.info("load model time elapsed:{}".format(end_time - start_time))
    return model, dataset


def load_model_and_dataset_retrain(train_config):
    start_time = time.perf_counter()
    dataset = load_dataset(train_config)
    if dataset is None:
        logger.error("dataset file not found")
        return None,None
    logger.info("feature info:{}".format(dataset.feature_info))
    latent_dim = train_config["latent_dim"]
    intermediate_dim = train_config["intermediate_dim"]
    model = CVAE(dataset.numeric_digits + dataset.categorical_digits, dataset.label_size, intermediate_dim, latent_dim,
                 dataset)
    model = load_torch_model(train_config, model)
    model.to(dataset.device)
    if model is None:
        logger.error("model file not found")
        return
    end_time = time.perf_counter()
    logger.info("load model time elapsed:{}".format(end_time - start_time))
    batch_size = train_config["batch_size"]
    epochs = train_config["inc_epochs"]
    lr = train_config["lr"]
    model = torch_cvae_train(model, dataset, learning_rate=lr, epochs=epochs, batch_size=batch_size)
    postfix = ''
    if 'inc_train_flag' in train_config:
        postfix = train_config['inc_train_flag']
    save_torch_model(model, train_config, postfix=postfix)
    save_dataset(dataset, train_config, postfix=postfix)
    return model, dataset

def generate_group_samples(sample_count, label, latent_dim, batch_size, model, z_decoded_list):
    start_time = time.perf_counter()
    while sample_count > 0:
        each_step_samples = sample_count if sample_count < batch_size else batch_size
        each_label = label[:each_step_samples, ]
        sample_count -= batch_size
        ### 0-1 normal
        mean = torch.zeros(each_step_samples, latent_dim).to(model.device)
        std = mean + 1

        ### param normal
        noise = torch.normal(mean=mean, std=std).to(model.device)
        # print("======noise.shape: ", noise.shape)
        # print("======each_label.shape: ", each_label.shape)
        fake = model.decode(noise, each_label)

        ### activate output
        column_list = []
        st = 0
        for digit, activ in model.output_info:
            ed = st + digit
            if activ == 'tanh':
                column_list.append(torch.tanh(fake[:, st:ed]))
            elif activ == 'softmax':
                column_list.append(torch.softmax(fake[:, st:ed], dim=1))
            elif activ == 'sigmoid':
                column_list.append(torch.sigmoid(fake[:, st:ed]))
            else:
                column_list.append(fake[:, st:ed])
                # column_list.append(torch.tanh(fake[:, st:ed]))
            st = ed
        fake = torch.cat(column_list, dim=1)
        z_decoded = fake.detach().cpu().numpy()
        z_decoded_list.append(z_decoded)
    end_time = time.perf_counter()
    # logger.info('generate group samples time:{}'.format(end_time - start_time))


def generate_samples_with_allocation(dataset, model, sample_allocation, sample_rates,
                                     train_config):
    start_time = time.perf_counter()
    batch_size = train_config["batch_size"]
    latent_dim = train_config["latent_dim"]
    categorical_encoding = train_config["categorical_encoding"]
    z_decoded = []
    label_value_mapping = dataset.label_value_mapping
    label_size = len(label_value_mapping)
    threads=[]

    for label_value_idx, label_value in label_value_mapping.items():
        if label_value in sample_allocation:
            sample_count = int(sample_allocation[label_value])
            if categorical_encoding == 'binary':
                mapping = dataset.label_mapping_out
                label = [mapping.loc[label_value_idx].values]
                # print("=====label: ",label)
                label = torch.from_numpy(np.repeat(label, batch_size, axis=0)).to(model.device)
                # label = np.tile(label, (sample_count, 1))
            else:
                label = np.ones((batch_size,)) * label_value_idx
                label = torch.from_numpy(to_categorical(label, label_size)).to(model.device)

            # print("=====label: ",label)
            # print("=====sample_count: ",sample_count)
            generate_group_samples(sample_count, label, latent_dim, batch_size, model, z_decoded)

    # for t in threads:
    #     t.join()

    # print("=====z_decoded: ",z_decoded)
    z_decoded = np.concatenate(z_decoded, axis=0)
    samples_df = dataset.decode_samples(z_decoded)
    # print("label_columns:",label_columns)
    # print("label_column_name:",dataset.label_column_name)
    # if len(label_columns)>1:
    #     samples_df[dataset.label_column_name]=samples_df[label_columns].astype(str).agg('-'.join, axis=1)
    # samples_df = samples_df.dropna(subset=['size'])
    # samples_df['size'] = samples_df['size'].astype(int)

    # for sql59
    # samples_df['ss_store_sk'] = samples_df['ss_store_sk'].apply(lambda x: np.random.choice(['1.0', '2.0', '4.0', '7.0', '8.0', '10.0']) if x == 'unused label' else x)
    # samples_df['ss_item_sk'] = samples_df['ss_item_sk'].apply(lambda x: np.random.choice(range(1, 18001)) if x == 'unused label' else x)

    samples_df=generate_label_column(samples_df,train_config['label_columns'],train_config['bucket_columns'],dataset.label_column_name)
    samples_df['{}_rate'.format(dataset.name)] = samples_df[dataset.label_column_name].map(sample_rates)

    # for sql27
    # samples_df['ss_store_sk'] = samples_df['ss_store_sk'].apply(lambda x: np.random.choice(['1.0', '2.0', '4.0', '7.0', '8.0', '10.0']) if x == 'unused label' else x)

    # print("========samples_df: ", samples_df[:50])
    
    # samples_df = pd.concat(samples)
    end_time = time.perf_counter()
    logger.info('sampling time:{}'.format(end_time - start_time))
    return samples_df

def generate_label_column(df, label_columns,bucket_columns,label_column_name):
        if len(bucket_columns)>0:
            for col in bucket_columns:
                df[col+"_bucket"]=(df[col]).mod(6)
                col_idx=label_columns.index(col)
                label_columns[col_idx]=label_columns[col_idx]+"_bucket"

        if len(label_columns) > 1:
            df[label_column_name] = df[label_columns].astype(str).agg('-'.join, axis=1)
        return df

def house_sampling(model, dataset, sample_rate):
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    sample_rates = {}
    sample_allocation = {}
    logger.info("house sampling rate:{}".format(sample_rate))
    # print("=======label_value_mapping: ", label_value_mapping)
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = round(label_count * sample_rate)
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / label_count
    return sample_allocation, sample_rates


def senate_sampling(model, dataset, sample_rate):
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate
    group_nums = len(label_group_counts)
    each_group_samples = int(total_samples / group_nums) + 1
    logger.info("senate sampling rate:{}".format(sample_rate))
    sample_allocation = {}
    sample_rates = {}
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = each_group_samples if each_group_samples < label_count else label_count
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / label_count
    return sample_allocation, sample_rates


# split the sample num into two part, one for senate, one for house
def advance_senate_sampling(model, dataset, sample_rate):
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    total_rows = dataset.total_rows
    group_nums = len(label_group_counts)
    left_out_rate = 0.7
    total_samples = total_rows * sample_rate
    each_group_samples = math.ceil(total_samples * left_out_rate / group_nums)
    rest = math.ceil(total_samples * (1 - left_out_rate))
    logger.info("advance senate sampling rate:{}".format(sample_rate))
    label_sample_counts = {}
    small_group_total_rows = 0
    for (label_value, count) in label_group_counts.items():
        if count > each_group_samples:
            label_sample_counts[label_value] = each_group_samples
        else:
            label_sample_counts[label_value] = label_group_counts[label_value]
            rest += each_group_samples - label_sample_counts[label_value]
            small_group_total_rows += label_group_counts[label_value]
    big_group_total_rows = total_rows - small_group_total_rows
    for (label_value, count) in label_group_counts.items():
        if count > each_group_samples:
            label_sample_counts[label_value] += math.ceil(
                rest * (label_group_counts[label_value] / big_group_total_rows))

    sample_allocation = {}
    sample_rates = {}
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        sample_count = label_sample_counts[label_value] if label_sample_counts[
                                                               label_value] < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / group_count
    return sample_allocation, sample_rates


def statistics_sampling(model, dataset, sample_rate, query_config):
    model.eval()
    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate
    logger.info("statistics sampling rate:{}".format(sample_rate))
    sample_allocation = {}
    sample_rates = {}
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        relative_variances = sum([label_group_relative_stds[col][label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(total_samples * (relative_variances / sum_relative_variance))
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates[label_value] = sample_count / group_count
    # logger.info("statistics sampling allocation:{}".format(sample_allocation))
    return sample_allocation, sample_rates


def statistics_sampling_with_small_group(model, dataset, sample_rate, query_config):
    model.eval()
    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate

    # statistics_sampling_samples = total_samples * 0.5
    statistics_sampling_samples = total_samples * 1.0
    small_group_sampling_samples = total_samples * 0.3
    small_group_K = small_group_sampling_samples / len(label_group_counts)

    sample_allocation = {}
    sample_rates = {}
    if small_group_K < 1:
        small_group_K = 1
    # count = 0
    # print("=====label_value_mapping: ", label_value_mapping)
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        relative_variances = sum([label_group_relative_stds[col][label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(statistics_sampling_samples * (relative_variances / sum_relative_variance))
        group_sample += small_group_K
        # if group_sample < 1:
        #     group_sample = 1
        #     count += 1
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        if group_count == 0:
            sample_rates[label_value] = 0.0
        else:
            sample_rates[label_value] = sample_count / group_count
        
    # print('======count: ', count)
    print('======small_group_K: ', small_group_K)
    return sample_allocation, sample_rates

def statistics_sampling_with_multi_label(model, dataset, sample_rate, query_config):
    model.eval()
    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    multi_label_group_beta = dataset.multi_label_group_beta
    multi_label_group_beta_sums = dataset.multi_label_group_beta_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate

    small_group_K = int(total_samples * 0.3 / len(label_group_counts))
    print('======small_group_K: ', small_group_K)
    statistics_sampling_samples = total_samples #- 132*small_group_K

    sample_allocation = {}
    sample_rates = {}
    count = 0
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        beta = sum([multi_label_group_beta[col][label_value] for col in numeric_columns])
        sum_beta = sum([multi_label_group_beta_sums[col] for col in numeric_columns])
        group_sample = int(statistics_sampling_samples * (beta / sum_beta))
        # group_sample += small_group_K
        if group_sample < small_group_K:
            group_sample = small_group_K
            count += 1
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        if group_count == 0:
            sample_rates[label_value] = 0.0
        else:
            sample_rates[label_value] = sample_count / group_count
        
    print('======count: ', count)
    return sample_allocation, sample_rates

def statistics_sampling_with_model_optimize_only_tow_label(model, dataset, sample_rate, query_config, train_config):
    model.eval()
    used_label = list(train_config['used_label'])
    # TODO：暂时只考虑两个属性
    all_label = [0, 1]
    unused_label = [i for i in all_label if i not in used_label]

    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    # TODO：暂时只考虑两个属性
    if unused_label[0] == 1:        # 使用第一个属性
        label_group_counts = dataset.label_group_counts_A
        label_value_mapping = dataset.label_value_mapping
        label_group_relative_stds = dataset.label_group_relative_stds_A
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums_A
    elif unused_label[0] == 0:       # 使用第二个属性
        label_group_counts = dataset.label_group_counts_B
        label_value_mapping = dataset.label_value_mapping
        label_group_relative_stds = dataset.label_group_relative_stds_B
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums_B


    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate

    # statistics_sampling_samples = total_samples * 0.5
    statistics_sampling_samples = total_samples * 1.0
    small_group_sampling_samples = total_samples * 0.3
    small_group_K = small_group_sampling_samples / len(label_group_counts)

    sample_allocation = {}
    sample_rates = {}
    if small_group_K < 1:
        small_group_K = 1
    # print("=====label_value_mapping: ", label_value_mapping)
    for label_value_idx, label_value in label_value_mapping.items():
        single_label_values = label_value.split('-')
        flag = False
        for unused in unused_label:         # 不相关的属性在查询中应该以 NaN 的形式出现，即label属性应该是 xxx-NaN
            if single_label_values[unused] != 'unused label':
                flag = True
                break
        if flag:
            continue
            
        # 只有 'xxx-NaN' 形式的 label_value 才能到这里来
        single_label_value = single_label_values[used_label[0]]  # 将 label_value 从 xxx-NaN 转换为 xxx; TODO：暂时只考虑两个属性
        group_count = label_group_counts[single_label_value]
        relative_variances = sum([label_group_relative_stds[col][single_label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(statistics_sampling_samples * (relative_variances / sum_relative_variance))
        group_sample += small_group_K
        sample_count = group_sample if group_sample < group_count else group_count
        # sample_allocation 和 sample_rates 的 key 还是 'xxx-NaN' 形式
        sample_allocation[label_value] = sample_count
        if group_count == 0:
            sample_rates[label_value] = 0.0
        else:
            sample_rates[label_value] = sample_count / group_count
        
    print('======small_group_K: ', small_group_K)
    return sample_allocation, sample_rates

def statistics_sampling_with_model_optimize(model, dataset, sample_rate, query_config, train_config):
    model.eval()
    used_label = list(train_config['used_label'])
    # print('used_label: ', used_label)
    # TODO：暂时只考虑四个属性
    all_label = [0, 1, 2, 3]
    unused_label = [i for i in all_label if i not in used_label]

    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    # TODO：暂时只考虑四个属性
    if len(used_label)==2 and used_label[0] == 0 and used_label[1] == 3:       # 使用第一、四个属性
        label_group_counts = dataset.label_group_counts_E
        label_group_relative_stds = dataset.label_group_relative_stds
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums
    elif used_label[0] == 0:        # 使用第一个属性
        label_group_counts = dataset.label_group_counts_A
        label_group_relative_stds = dataset.label_group_relative_stds_A
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums_A
    elif used_label[0] == 1:       # 使用第二个属性
        label_group_counts = dataset.label_group_counts_B
        label_group_relative_stds = dataset.label_group_relative_stds_B
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums_B
    elif used_label[0] == 2:       # 使用第三个属性
        label_group_counts = dataset.label_group_counts_C
        label_group_relative_stds = dataset.label_group_relative_stds_C
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums_C
    elif used_label[0] == 3:       # 使用第四个属性
        label_group_counts = dataset.label_group_counts_D
        label_group_relative_stds = dataset.label_group_relative_stds_D
        label_group_relative_stds_sums = dataset.label_group_relative_stds_sums_D


    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate

    # statistics_sampling_samples = total_samples * 0.5
    statistics_sampling_samples = total_samples * 1.0
    small_group_sampling_samples = total_samples * 0.3
    small_group_K = small_group_sampling_samples / len(label_group_counts)

    sample_allocation = {}
    sample_rates = {}
    small_group_K += 1
    # if small_group_K < 1:
    #     small_group_K = 1
    # print("=====label_value_mapping: ", label_value_mapping)
    for label_value_idx, label_value in label_value_mapping.items():
        single_label_values = label_value.split('-')
        flag = False
        for unused in unused_label:         # 不相关的属性在查询中应该以 NaN 的形式出现，即label属性应该是 xxx-NaN
            if single_label_values[unused] != 'unused label':
                flag = True
                break
        if flag:
            continue
            
        # 只有 'xxx-NaN' 形式的 label_value 才能到这里来
        single_label_value = single_label_values[used_label[0]]  # 将 label_value 从 xxx-NaN 转换为 xxx; TODO：暂时只考虑四个属性
        if len(used_label) == 2:
            if single_label_values[used_label[0]] == 'unused label' or single_label_values[used_label[1]] == 'unused label':    # 如果有两个label属性，那么都不能为'unused label'
                continue
            single_label_value = "-".join([single_label_values[used_label[0]], single_label_values[used_label[1]]])   # for sql59
        group_count = label_group_counts[single_label_value]
        relative_variances = sum([label_group_relative_stds[col][single_label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(statistics_sampling_samples * (relative_variances / sum_relative_variance))
        group_sample += small_group_K
        sample_count = group_sample if group_sample < group_count else group_count
        # sample_allocation 和 sample_rates 的 key 还是 'xxx-NaN' 形式
        sample_allocation[label_value] = sample_count
        if group_count == 0:
            sample_rates[label_value] = 0.0
        else:
            sample_rates[label_value] = sample_count / group_count
        
    print('======small_group_K: ', small_group_K)
    return sample_allocation, sample_rates

def save_samples(samples, train_config):
    samples_name = "{}_{}_{}_ld{}_id{}_bs{}_ep{}_rate{}_{}_{}.csv".format(train_config["model_type"], train_config["name"],
                                                                          '_'.join(train_config["label_columns"]),
                                                                          train_config["latent_dim"],
                                                                          train_config["intermediate_dim"],
                                                                          train_config["batch_size"],
                                                                          train_config["epochs"], train_config["sample_rate"],
                                                                          train_config["categorical_encoding"],
                                                                          (train_config["numeric_encoding"] + str(
                                                                              train_config["max_clusters"])) if train_config[
                                                                                                             "numeric_encoding"] == 'gaussian' else
                                                                          train_config["numeric_encoding"])
    samples.to_csv("./output/{}".format(samples_name), index=False)

def read_samples(train_config):
    samples_name = "{}_{}_{}_ld{}_id{}_bs{}_ep{}_rate{}_{}_{}.csv".format(train_config["model_type"], train_config["name"],
                                                                          '_'.join(train_config["label_columns"]),
                                                                          train_config["latent_dim"],
                                                                          train_config["intermediate_dim"],
                                                                          train_config["batch_size"],
                                                                          train_config["epochs"], train_config["sample_rate"],
                                                                          train_config["categorical_encoding"],
                                                                          (train_config["numeric_encoding"] + str(
                                                                              train_config["max_clusters"])) if train_config[
                                                                                                             "numeric_encoding"] == 'gaussian' else
                                                                          train_config["numeric_encoding"])
    return pd.read_csv("./output/{}".format(samples_name))
