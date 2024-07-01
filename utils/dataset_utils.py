# GiG
import math
import time
import pickle
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import logging

from utils.binary_encoder import BinaryEncoder
from utils.gaussian_encoder import GaussianEncoder
logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    def __init__(self, param):
        self.gpu_num = 0
        if 'gpu_num' in param:
            self.gpu_num = param['gpu_num']
        self.device = torch.device("cuda:{}".format(self.gpu_num) if torch.cuda.is_available() else "cpu")
        # self.load_data(param)
        self.inc_train_flag = 'origin_train'
        self.categorical_encoding = param['categorical_encoding']
        self.numeric_encoding = param['numeric_encoding']
        self.sample_for_train = param['sample_for_train']
        self.load_data(param)
        self.delete_data = False
        encoded_categorical = None
        encoded_numeric = None
        start_time = time.perf_counter()

        if len(self.all_categorical_columns) > 0:
            categorical_data = self.origin_df[self.all_categorical_columns]
            if self.categorical_encoding == 'binary':
                encoded_categorical = self.encode_categorical_data_binary(categorical_data)
                self.label = self.encode_label_binary(self.label_df[[self.label_column_name]])
            else:
                encoded_categorical = self.encode_categorical_data_one_hot(categorical_data)
                self.label = self.encode_label_binary(self.label_df[[self.label_column_name]])
        end_time = time.perf_counter()
        logger.info('encode categorical columns time elapsed:{}'.format(end_time - start_time))

        start_time = time.perf_counter()
        if len(self.numeric_columns) > 0:
            numeric_data = self.origin_df[self.numeric_columns]
            if self.numeric_encoding == 'gaussian':
                self.gaussian_max_clusters = param["max_clusters"]
                encoded_numeric = self.encode_numeric_data_gaussian(numeric_data)
            elif self.numeric_encoding == 'stdmm':
                encoded_numeric = self.encode_numeric_data_stdmm(numeric_data)
            else:
                encoded_numeric = self.encode_numeric_data_mm(numeric_data)
        end_time = time.perf_counter()
        logger.info('encode numeric columnstime elapsed:{}'.format(end_time - start_time))

        if encoded_categorical is None:
            self.data = encoded_numeric
        elif encoded_numeric is None:
            self.data = encoded_categorical
        else:
            self.data = pd.concat([encoded_numeric.reset_index(drop=True), encoded_categorical.reset_index(drop=True)],
                                  axis=1)

        if 'multi_label' in param and param['multi_label'] == 'true':       # 使用扩展的针对两个标签属性的分配算法
            label_columns_copy = param['label_columns'].copy()
            self.label_column_name_A = label_columns_copy[0]
            self.label_column_name_B = label_columns_copy[1]
            self.label_df[self.label_column_name_A] = self.label_df[self.label_column_name_A].astype(str)
            self.label_df[self.label_column_name_B] = self.label_df[self.label_column_name_B].astype(str)

            # 记录根据 A 属性进行分组的统计信息，TODO 实际上好像只要count和mean，后面可以精简下
            self.label_group_stds_A = {}
            self.label_group_means_A = {}
            self.label_group_relative_stds_A = {}
            self.label_group_relative_stds_sums_A = {}
            self.label_group_counts_A = self.label_df[self.label_column_name_A].value_counts().to_dict()
            
            # 记录根据 B 属性进行分组的统计信息
            self.label_group_stds_B = {}
            self.label_group_means_B = {}
            self.label_group_relative_stds_B = {}
            self.label_group_relative_stds_sums_B = {}
            self.label_group_counts_B = self.label_df[self.label_column_name_B].value_counts().to_dict()

            # 记录根据 C 属性进行分组的统计信息
            self.label_group_stds = {}
            self.label_group_means = {}
            self.label_group_relative_stds = {}
            self.label_group_relative_stds_sums = {}

            # 用于保存 C 到 A、B 的映射，对应公式中的 f1(c) 和 f2(c)
            self.map_C_to_A = {}
            self.map_C_to_B = {}
            for ccc in self.label_df[self.label_column_name].unique().tolist():
            # for ccc in self.label_group_stds[str(self.numeric_columns[0])]:      # 好像这样也能遍历 key
                self.map_C_to_A[ccc] = ccc.split('-')[0]
                self.map_C_to_B[ccc] = ccc.split('-')[1]

            self.multi_label_group_beta = {}
            self.multi_label_group_beta_sums = {}

            for col in self.numeric_columns:
                # 计算根据 A 属性进行分组的统计信息
                self.label_group_stds_A[col] = self.label_df.groupby(self.label_column_name_A)[col].std(ddof=0).to_dict()
                self.label_group_means_A[col] = self.label_df.groupby(self.label_column_name_A)[col].mean().to_dict()
                self.label_group_relative_stds_A[col] = { k: self.label_group_stds_A[col][k] / self.label_group_means_A[col][k] 
                    if self.label_group_means_A[col][k] != 0 else 0 for k in self.label_group_stds_A[col]}
                self.label_group_relative_stds_sums_A[col] = sum(self.label_group_relative_stds_A[col].values())

                # 计算根据 B 属性进行分组的统计信息
                self.label_group_stds_B[col] = self.label_df.groupby(self.label_column_name_B)[col].std(ddof=0).to_dict()
                self.label_group_means_B[col] = self.label_df.groupby(self.label_column_name_B)[col].mean().to_dict()
                self.label_group_relative_stds_B[col] = { k: self.label_group_stds_B[col][k] / self.label_group_means_B[col][k] 
                    if self.label_group_means_B[col][k] != 0 else 0 for k in self.label_group_stds_B[col]}
                self.label_group_relative_stds_sums_B[col] = sum(self.label_group_relative_stds_B[col].values())

                # 计算根据 C 属性进行分组的统计信息
                self.label_group_stds[col] = self.label_df.groupby(self.label_column_name)[col].std(ddof=0).to_dict()
                self.label_group_means[col] = self.label_df.groupby(self.label_column_name)[col].mean().to_dict()
                self.label_group_relative_stds[col] = { k: self.label_group_stds[col][k] / self.label_group_means[col][k] 
                    if self.label_group_means[col][k] != 0 else 0 for k in self.label_group_stds[col]}
                self.label_group_relative_stds_sums[col] = sum(self.label_group_relative_stds[col].values())

                self.multi_label_group_beta[col] = { k: self.label_group_counts[k] * self.label_group_stds[col][k] * 
                    math.sqrt(1 / (self.label_group_counts_A[self.map_C_to_A[k]]**2 * self.label_group_means_A[col][self.map_C_to_A[k]]**2) + 
                              1 / (self.label_group_counts_B[self.map_C_to_B[k]]**2 * self.label_group_means_B[col][self.map_C_to_B[k]]**2)) 
                    for k in self.label_group_stds[col]}
                self.multi_label_group_beta_sums[col] = sum(self.multi_label_group_beta[col].values())
            self.label_size = self.label.shape[1]
            
            # # 用于保存 C 到 A、B 的映射，对应公式中的 f1(c) 和 f2(c)
            # self.map_C_to_A = {}
            # self.map_C_to_B = {}
            # # for ccc in self.label_df[self.label_column_name].unique().tolist():
            # for ccc in self.label_group_stds[self.numeric_columns[0]]:      # 好像这样也能遍历 key
            #     self.map_C_to_A[ccc] = ccc.split('-')[0]
            #     self.map_C_to_B[ccc] = ccc.split('-')[1]

            # self.multi_label_group_beta = {}
            # self.multi_label_group_beta_sums = {}
            # for col in self.numeric_columns:
            #     self.multi_label_group_beta[col] = { k: self.label_group_counts[col][k] * self.label_group_stds[col][k] * 
            #         math.sqrt(1 / (self.label_group_counts_A[self.map_C_to_A[k]]**2 * self.label_group_means_A[self.map_C_to_A[k]]**2) + 
            #                   1 / (self.label_group_counts_B[self.map_C_to_B[k]]**2 * self.label_group_means_B[self.map_C_to_B[k]]**2))
            #         for k in self.label_group_stds[col]}
            #     self.multi_label_group_beta_sums[col] = sum(self.multi_label_group_beta[col].values())
        elif 'used_label' in param:       # 使用扩展的针对模型数量优化的分配算法，暂时只考虑两个属性 or 四个属性（针对两个实验）
            label_columns_copy = param['label_columns'].copy()
            if len(label_columns_copy) == 4:        # 四个属性，tpcds_whole实验
                # A, B, C, D 依次为 ss_sold_date_sk, ss_item_sk, ss_customer_sk, ss_store_sk
                self.label_column_name_A = label_columns_copy[0]
                self.label_column_name_B = label_columns_copy[1]
                self.label_column_name_C = label_columns_copy[2]
                self.label_column_name_D = label_columns_copy[3]
                self.label_df[self.label_column_name_A] = self.label_df[self.label_column_name_A].astype(str)
                self.label_df[self.label_column_name_B] = self.label_df[self.label_column_name_B].astype(str)
                self.label_df[self.label_column_name_C] = self.label_df[self.label_column_name_C].astype(str)
                self.label_df[self.label_column_name_D] = self.label_df[self.label_column_name_D].astype(str)

                # 记录根据 A 属性进行分组的统计信息，TODO 实际上好像只要count和mean，后面可以精简下
                self.label_group_stds_A = {}
                self.label_group_means_A = {}
                self.label_group_relative_stds_A = {}
                self.label_group_relative_stds_sums_A = {}
                self.label_group_counts_A = self.label_df[self.label_column_name_A].value_counts().to_dict()
                
                # 记录根据 B 属性进行分组的统计信息
                self.label_group_stds_B = {}
                self.label_group_means_B = {}
                self.label_group_relative_stds_B = {}
                self.label_group_relative_stds_sums_B = {}
                self.label_group_counts_B = self.label_df[self.label_column_name_B].value_counts().to_dict()

                # 记录根据 C 属性进行分组的统计信息
                self.label_group_stds_C = {}
                self.label_group_means_C = {}
                self.label_group_relative_stds_C = {}
                self.label_group_relative_stds_sums_C = {}
                self.label_group_counts_C = self.label_df[self.label_column_name_C].value_counts().to_dict()
                
                # 记录根据 D 属性进行分组的统计信息
                self.label_group_stds_D = {}
                self.label_group_means_D = {}
                self.label_group_relative_stds_D = {}
                self.label_group_relative_stds_sums_D = {}
                self.label_group_counts_D = self.label_df[self.label_column_name_D].value_counts().to_dict()

                # 记录根据 E 属性进行分组的统计信息 (A-D, 即 ss_sold_date_sk-ss_store_sk)
                self.label_group_stds = {}
                self.label_group_means = {}
                self.label_group_relative_stds = {}
                self.label_group_relative_stds_sums = {}

                self.label_column_name_E = "-".join([label_columns_copy[0], label_columns_copy[3]])
                self.label_df[self.label_column_name_E] = self.label_df[[label_columns_copy[0], label_columns_copy[3]]].astype(str).agg('-'.join, axis=1)
                self.label_group_counts_E = self.label_df[self.label_column_name_E].value_counts().to_dict()

                for col in self.numeric_columns:
                    # 计算根据 A 属性进行分组的统计信息
                    self.label_group_stds_A[col] = self.label_df.groupby(self.label_column_name_A)[col].std(ddof=0).to_dict()
                    self.label_group_means_A[col] = self.label_df.groupby(self.label_column_name_A)[col].mean().to_dict()
                    self.label_group_relative_stds_A[col] = { k: self.label_group_stds_A[col][k] / self.label_group_means_A[col][k] 
                        if self.label_group_means_A[col][k] != 0 else 0 for k in self.label_group_stds_A[col]}
                    self.label_group_relative_stds_sums_A[col] = sum(self.label_group_relative_stds_A[col].values())

                    # 计算根据 B 属性进行分组的统计信息
                    self.label_group_stds_B[col] = self.label_df.groupby(self.label_column_name_B)[col].std(ddof=0).to_dict()
                    self.label_group_means_B[col] = self.label_df.groupby(self.label_column_name_B)[col].mean().to_dict()
                    self.label_group_relative_stds_B[col] = { k: self.label_group_stds_B[col][k] / self.label_group_means_B[col][k] 
                        if self.label_group_means_B[col][k] != 0 else 0 for k in self.label_group_stds_B[col]}
                    self.label_group_relative_stds_sums_B[col] = sum(self.label_group_relative_stds_B[col].values())

                    # 计算根据 C 属性进行分组的统计信息
                    self.label_group_stds_C[col] = self.label_df.groupby(self.label_column_name_C)[col].std(ddof=0).to_dict()
                    self.label_group_means_C[col] = self.label_df.groupby(self.label_column_name_C)[col].mean().to_dict()
                    self.label_group_relative_stds_C[col] = { k: self.label_group_stds_C[col][k] / self.label_group_means_C[col][k] 
                        if self.label_group_means_C[col][k] != 0 else 0 for k in self.label_group_stds_C[col]}
                    self.label_group_relative_stds_sums_C[col] = sum(self.label_group_relative_stds_C[col].values())

                    # 计算根据 D 属性进行分组的统计信息
                    self.label_group_stds_D[col] = self.label_df.groupby(self.label_column_name_D)[col].std(ddof=0).to_dict()
                    self.label_group_means_D[col] = self.label_df.groupby(self.label_column_name_D)[col].mean().to_dict()
                    self.label_group_relative_stds_D[col] = { k: self.label_group_stds_D[col][k] / self.label_group_means_D[col][k] 
                        if self.label_group_means_D[col][k] != 0 else 0 for k in self.label_group_stds_D[col]}
                    self.label_group_relative_stds_sums_D[col] = sum(self.label_group_relative_stds_D[col].values())

                    # 计算根据 E(A-D) 属性进行分组的统计信息
                    self.label_group_stds[col] = self.label_df.groupby(self.label_column_name_E)[col].std(ddof=0).to_dict()
                    self.label_group_means[col] = self.label_df.groupby(self.label_column_name_E)[col].mean().to_dict()
                    self.label_group_relative_stds[col] = { k: self.label_group_stds[col][k] / self.label_group_means[col][k] 
                        if self.label_group_means[col][k] != 0 else 0 for k in self.label_group_stds[col]}
                    self.label_group_relative_stds_sums[col] = sum(self.label_group_relative_stds[col].values())

                self.label_size = self.label.shape[1]
            else:
                self.label_column_name_A = label_columns_copy[0]
                self.label_column_name_B = label_columns_copy[1]
                self.label_df[self.label_column_name_A] = self.label_df[self.label_column_name_A].astype(str)
                self.label_df[self.label_column_name_B] = self.label_df[self.label_column_name_B].astype(str)

                # 记录根据 A 属性进行分组的统计信息，TODO 实际上好像只要count和mean，后面可以精简下
                self.label_group_stds_A = {}
                self.label_group_means_A = {}
                self.label_group_relative_stds_A = {}
                self.label_group_relative_stds_sums_A = {}
                self.label_group_counts_A = self.label_df[self.label_column_name_A].value_counts().to_dict()
                
                # 记录根据 B 属性进行分组的统计信息
                self.label_group_stds_B = {}
                self.label_group_means_B = {}
                self.label_group_relative_stds_B = {}
                self.label_group_relative_stds_sums_B = {}
                self.label_group_counts_B = self.label_df[self.label_column_name_B].value_counts().to_dict()

                # 记录根据 C 属性进行分组的统计信息
                self.label_group_stds = {}
                self.label_group_means = {}
                self.label_group_relative_stds = {}
                self.label_group_relative_stds_sums = {}

                for col in self.numeric_columns:
                    # 计算根据 A 属性进行分组的统计信息
                    self.label_group_stds_A[col] = self.label_df.groupby(self.label_column_name_A)[col].std(ddof=0).to_dict()
                    self.label_group_means_A[col] = self.label_df.groupby(self.label_column_name_A)[col].mean().to_dict()
                    self.label_group_relative_stds_A[col] = { k: self.label_group_stds_A[col][k] / self.label_group_means_A[col][k] 
                        if self.label_group_means_A[col][k] != 0 else 0 for k in self.label_group_stds_A[col]}
                    self.label_group_relative_stds_sums_A[col] = sum(self.label_group_relative_stds_A[col].values())

                    # 计算根据 B 属性进行分组的统计信息
                    self.label_group_stds_B[col] = self.label_df.groupby(self.label_column_name_B)[col].std(ddof=0).to_dict()
                    self.label_group_means_B[col] = self.label_df.groupby(self.label_column_name_B)[col].mean().to_dict()
                    self.label_group_relative_stds_B[col] = { k: self.label_group_stds_B[col][k] / self.label_group_means_B[col][k] 
                        if self.label_group_means_B[col][k] != 0 else 0 for k in self.label_group_stds_B[col]}
                    self.label_group_relative_stds_sums_B[col] = sum(self.label_group_relative_stds_B[col].values())

                    # 计算根据 C 属性进行分组的统计信息
                    self.label_group_stds[col] = self.label_df.groupby(self.label_column_name)[col].std(ddof=0).to_dict()
                    self.label_group_means[col] = self.label_df.groupby(self.label_column_name)[col].mean().to_dict()
                    self.label_group_relative_stds[col] = { k: self.label_group_stds[col][k] / self.label_group_means[col][k] 
                        if self.label_group_means[col][k] != 0 else 0 for k in self.label_group_stds[col]}
                    self.label_group_relative_stds_sums[col] = sum(self.label_group_relative_stds[col].values())

                self.label_size = self.label.shape[1]

        elif self.label_column_name is not None:
            # collect variance and mean of each numeric columns for each label group
            self.label_group_stds = {}
            self.label_group_means = {}
            self.label_group_relative_stds = {}
            self.label_group_relative_stds_sums = {}
            self.label_group_relative_stds_with_num = {}
            self.label_group_relative_stds_with_num_sums = {}
            for col in self.numeric_columns:
                self.label_group_stds[col] = self.label_df.groupby(self.label_column_name)[col].std(ddof=0).to_dict()
                self.label_group_means[col] = self.label_df.groupby(self.label_column_name)[col].mean().to_dict()

                # print("==========self.label_column_name: ",self.label_column_name)
                # print("==========self.label_group_stds[col].size: ", len(self.label_group_stds[col]))
                # print("==========self.label_group_means[col].size: ", len(self.label_group_means[col]))
                # print("==========self.label_group_stds: ",self.label_group_stds)
                # print("==========self.label_group_means: ",self.label_group_means)
                # print("==========self.origin_df.isnull().sum(): ",self.origin_df.isnull().sum())
                self.label_group_relative_stds[col] = {
                    k: self.label_group_stds[col][k] / self.label_group_means[col][k] if self.label_group_means[col][
                                                                                             k] != 0 else 0
                    for k in self.label_group_stds[col]}

                self.label_group_relative_stds_sums[col] = sum(self.label_group_relative_stds[col].values())

                # print("==========label_group_relative_stds: ",self.label_group_relative_stds)
                # print("==========label_group_relative_stds_sums: ",self.label_group_relative_stds_sums)


                self.label_group_relative_stds_with_num[col] = {
                    k: self.label_group_relative_stds[col][k] * math.sqrt(self.label_group_counts[k])
                    for k in self.label_group_stds[col]}
                self.label_group_relative_stds_with_num_sums[col] = sum(
                    self.label_group_relative_stds_with_num[col].values())

            # self.label = self.data.iloc[:, self.data.columns.str.contains(self.label_column_name)]
            self.label_size = self.label.shape[1]
        else:
            self.label = pd.DataFrame(np.zeros((self.total_rows, 1)))
            self.label_size = 0
        self.feature_data=self.data
        # if self.label_column_name not in self.all_columns:
        #     self.feature_data.drop([t for t in self.feature_data.columns if self.label_column_name in t],axis=1,inplace=True)
        self.data_dim = self.feature_data.shape[1]
        logger.info("data shape:{}".format(self.feature_data.shape))

        self.numeric_digits = 0
        self.categorical_digits = 0
        self.feature_info = []
        self.encoded_output_info = []
        for col_name in self.all_columns:
            if col_name in self.all_categorical_columns:
                digit = self.column_digits[col_name]
                self.feature_info.append((col_name, "categorical", digit))
                if self.categorical_encoding == 'binary':
                    self.encoded_output_info.append((digit, 'sigmoid'))
                else:
                    self.encoded_output_info.append((digit, 'softmax'))
                self.categorical_digits += digit
            else:
                if self.numeric_encoding == 'gaussian':
                    digit = self.gme.gms[col_name].num_components
                    self.feature_info.append((col_name, "numerical", digit + 1))
                    # self.encoded_output_info.append((1, 'tanh'))
                    self.encoded_output_info.append((1, 'no'))
                    self.encoded_output_info.append((digit, 'softmax'))
                    self.numeric_digits += digit + 1
                else:
                    self.feature_info.append((col_name, "numerical", 1))
                    # self.encoded_output_info.append((1, 'tanh'))
                    self.encoded_output_info.append((1, 'no'))
                    self.numeric_digits += 1

        logger.info('feature info:{}'.format(self.feature_info))
        logger.info("output info list:{}".format(self.encoded_output_info))
        self.raw_data = torch.from_numpy(self.feature_data.values.astype("float32")).to(self.device)
        self.raw_label_data = torch.from_numpy(self.label.values.astype("float32")).to(self.device)
        logger.info('load data successfully')

    def __getitem__(self, index):
        # return torch.from_numpy(self.raw_df[index, :])
        if self.inc_train_flag == 'inc_train':
            return self.inc_raw_data[index, :], self.inc_raw_label_data[index, :]
        elif self.inc_train_flag == 'sample_train':
            return self.inc_sample_raw_data[index, :], self.inc_sample_raw_label_data[index, :]
        elif self.inc_train_flag == 'old_train':
            return self.inc_old_raw_data[index, :], self.inc_old_raw_label_data[index, :]
        return self.raw_data[index, :], self.raw_label_data[index, :]

    def __len__(self):
        if self.inc_train_flag == 'inc_train':
            return self.inc_rows
        elif self.inc_train_flag == 'sample_train':
            return self.inc_sample_rows
        elif self.inc_train_flag == 'old_train':
            return self.inc_old_rows
        return self.total_rows

    def load_incremental_data(self, train_config):
        start_time = time.perf_counter()
        delimiter = train_config["delimiter"]
        filename = train_config["inc_data"]
        header = train_config["header"]
        if header == 1:
            df = pd.read_csv(filename, delimiter=delimiter)
        else:
            df = pd.read_csv(filename, header=None, delimiter=delimiter)

        self.delete_data = False       # 判断是否是删除数据
        if "left_data" in train_config:
            self.delete_data = True

        self.inc_df = df[self.all_columns]  # .copy(deep=True)
        self.inc_label_df = self.inc_df.copy(deep=True)
        # print("label group counts before:{}".format(self.label_group_counts))
        self.inc_rows = len(df)

        label_columns = train_config['label_columns']
        if len(label_columns) > 1:
            self.inc_label_df[self.label_column_name] = self.inc_label_df[label_columns].astype(str).agg('-'.join, axis=1)
        if self.label_column_name != None:
            self.inc_label_group_counts = self.inc_label_df[self.label_column_name].value_counts().to_dict()
            for label in self.inc_label_group_counts:
                if label in self.label_group_counts:
                    self.label_group_counts[label] += self.inc_label_group_counts[label]
                else:
                    self.label_group_counts[label] = self.inc_label_group_counts[label]
        # print("label group counts after:{}".format(self.label_group_counts))
        encoded_categorical = None
        encoded_numeric = None
        start_cat_time = time.perf_counter()
        if len(self.all_categorical_columns) > 0:
            categorical_data = self.inc_df[self.all_categorical_columns]
            if self.categorical_encoding == 'binary':
                encoded_categorical = self.bce.transform(categorical_data)
                self.inc_label = self.encode_label_binary(self.inc_label_df[[self.label_column_name]])
            else:
                onehot_encoded = self.ohe.transform(categorical_data).todense()
                encoded_categorical = pd.DataFrame(onehot_encoded, columns=self.onehot_encoded_columns)
                self.inc_label = self.encode_label_binary(self.inc_label_df[[self.label_column_name]])
        end_cat_time = time.perf_counter()
        logger.info('encode incremental data categorical columns time elapsed:{}'.format(end_cat_time - start_cat_time))

        start_num_time = time.perf_counter()
        if len(self.numeric_columns) > 0:
            numeric_data = self.inc_df[self.numeric_columns]
            if self.numeric_encoding == 'gaussian':
                encoded_numeric = self.gme.transform(numeric_data)
            elif self.numeric_encoding == 'stdmm':
                numeric_data = np.array(numeric_data)
                numeric_data = self.std_scaler.transform(numeric_data)
                numeric_data = self.mm_scaler.transform(numeric_data)
                encoded_numeric = pd.DataFrame(numeric_data, columns=self.numeric_columns)
            else:
                numeric_data = np.array(numeric_data)
                numeric_data = self.mm_scaler.transform(numeric_data)
                encoded_numeric = pd.DataFrame(numeric_data, columns=self.numeric_columns)
        end_num_time = time.perf_counter()
        logger.info('encode incremental data numeric columnstime elapsed:{}'.format(end_num_time - start_num_time))

        ### strategy one: use only incremental data to train
        if encoded_categorical is None:
            self.inc_data = encoded_numeric
        elif encoded_numeric is None:
            self.inc_data = encoded_categorical
        else:
            self.inc_data = pd.concat([encoded_numeric, encoded_categorical], axis=1)

        self.inc_raw_data = torch.from_numpy(self.inc_data.values.astype("float32")).to(self.device)
        self.inc_raw_label_data = torch.from_numpy(self.inc_label.values.astype("float32")).to(self.device)

        ### strategy two: use all incremental data and old data to train
        self.inc_old_data = pd.concat([self.inc_data, self.data], axis=0)
        self.inc_old_rows = len(self.inc_old_data)

        self.inc_old_label = pd.concat([self.inc_label, self.label], axis=0)
        self.inc_old_raw_data = torch.from_numpy(self.inc_old_data.values.astype("float32")).to(self.device)
        self.inc_old_raw_label_data = torch.from_numpy(self.inc_old_label.values.astype("float32")).to(self.device)


        ### strategy three:  use sample of incremental data and old data to train
        self.inc_sample_data = self.inc_old_data.sample(frac=0.2)
        # if self.label_column_name is not None:        # 这样好像无法处理多个标签属性，换成下面的两行
        #     self.inc_sample_label = self.inc_sample_data.iloc[:,
        #                             self.inc_sample_data.columns.str.contains(self.label_column_name)]
        # else:
        #     self.inc_sample_label = pd.DataFrame(np.zeros((self.inc_sample_rows, 1)))

        inc_sample_data_indices = [idx for idx in self.inc_sample_data.index]
        self.inc_sample_label = self.inc_old_label.iloc[inc_sample_data_indices, :]
        
        if "left_data" in train_config:         # 存在"left_data"这一项，表示进行数据删除，即 self.delete_data == True
            left_data_filename = train_config["left_data"]
            left_data_df = pd.read_csv(left_data_filename, delimiter=delimiter)
            
            self.left_df = left_data_df[self.all_columns]  # .copy(deep=True)
            self.left_label_df = self.left_df.copy(deep=True)
            self.left_rows = len(df)

            if len(label_columns) > 1:
                self.left_label_df[self.label_column_name] = self.left_label_df[label_columns].astype(str).agg('-'.join, axis=1)
            if self.label_column_name != None:
                # self.inc_label_group_counts = self.inc_label_df[self.label_column_name].value_counts().to_dict()
                for label in self.inc_label_group_counts:     # self.label_group_counts 应该减去删除的数量
                    if label in self.label_group_counts:
                        self.label_group_counts[label] -= self.inc_label_group_counts[label]*2      # 因为前面把inc额外加了一次，所以这里要减去两次
                    else:
                        self.label_group_counts[label] = self.inc_label_group_counts[label]
            encoded_categorical = None
            encoded_numeric = None
            if len(self.all_categorical_columns) > 0:
                categorical_data = self.left_df[self.all_categorical_columns]
                if self.categorical_encoding == 'binary':
                    encoded_categorical = self.bce.transform(categorical_data)
                    self.left_label = self.encode_label_binary(self.left_label_df[[self.label_column_name]])
                else:
                    onehot_encoded = self.ohe.transform(categorical_data).todense()
                    encoded_categorical = pd.DataFrame(onehot_encoded, columns=self.onehot_encoded_columns)
                    self.left_label = self.encode_label_binary(self.left_label_df[[self.label_column_name]])

            if len(self.numeric_columns) > 0:
                numeric_data = self.left_df[self.numeric_columns]
                if self.numeric_encoding == 'gaussian':
                    encoded_numeric = self.gme.transform(numeric_data)
                elif self.numeric_encoding == 'stdmm':
                    numeric_data = np.array(numeric_data)
                    numeric_data = self.std_scaler.transform(numeric_data)
                    numeric_data = self.mm_scaler.transform(numeric_data)
                    encoded_numeric = pd.DataFrame(numeric_data, columns=self.numeric_columns)
                else:
                    numeric_data = np.array(numeric_data)
                    numeric_data = self.mm_scaler.transform(numeric_data)
                    encoded_numeric = pd.DataFrame(numeric_data, columns=self.numeric_columns)

            if encoded_categorical is None:
                self.left_data = encoded_numeric
            elif encoded_numeric is None:
                self.left_data = encoded_categorical
            else:
                self.left_data = pd.concat([encoded_numeric, encoded_categorical], axis=1)

            # self.inc_sample_data = pd.concat([self.inc_data, self.left_data.sample(frac=0.003)], axis=0)    # test1
            # sampled_left_data = self.left_data.sample(frac=0.2)     # tpch-cn, tpcds-sw
            sampled_left_data = self.left_data.sample(frac=0.7)
            # self.inc_sample_data = pd.concat([self.inc_data.head(4000), sampled_left_data], axis=0)    # census2
            self.inc_sample_data = pd.concat([self.inc_data, sampled_left_data], axis=0)

            sampled_indices = [idx for idx in sampled_left_data.index]
            sampled_left_label = self.left_label.iloc[sampled_indices, :]
            # self.inc_sample_label = pd.concat([self.inc_label.head(4000), sampled_left_label], axis=0)  # census2
            self.inc_sample_label = pd.concat([self.inc_label, sampled_left_label], axis=0)

            # label_group_counts, label_value_mapping, label_group_relative_stds, label_group_relative_stds_sums, total_rows 后面要用到，需要更新
            # label_group_counts 前面已经更新，label_value_mapping 不用更新
            self.total_rows = len(self.left_df)
            self.label_group_stds = {}
            self.label_group_means = {}
            self.label_group_relative_stds = {}
            self.label_group_relative_stds_sums = {}
            for col in self.numeric_columns:
                self.label_group_stds[col] = self.left_label_df.groupby(self.label_column_name)[col].std(ddof=0).to_dict()
                self.label_group_means[col] = self.left_label_df.groupby(self.label_column_name)[col].mean().to_dict()
                self.label_group_relative_stds[col] = {
                    k: self.label_group_stds[col][k] / self.label_group_means[col][k] if self.label_group_means[col][k] != 0 else 0
                    for k in self.label_group_stds[col]}
                self.label_group_relative_stds_sums[col] = sum(self.label_group_relative_stds[col].values())

    
        self.inc_sample_rows = len(self.inc_sample_data)

        # if self.label_column_name is not None: 
        #     self.inc_sample_label = self.inc_sample_data.iloc[:,
        #                             self.inc_sample_data.columns.str.contains(self.label_column_name)]
        # else:
        #     self.inc_sample_label = pd.DataFrame(np.zeros((self.inc_sample_rows, 1)))

        self.inc_sample_raw_data = torch.from_numpy(self.inc_sample_data.values.astype("float32")).to(self.device)
        self.inc_sample_raw_label_data = torch.from_numpy(self.inc_sample_label.values.astype("float32")).to(
            self.device)

        # self.raw_data = torch.cat((self.raw_data, self.inc_raw_data), dim=0)
        # self.raw_label_data = torch.cat((self.label, self.inc_raw_label_data), dim=0)
        end_time = time.perf_counter()
        logger.info('load incremental data time elapsed:{}'.format(end_time - start_time))

    def filter_outlier(self):
        numeric_data = self.origin_df[self.numeric_columns]
        # stds = numeric_data.std()
        # means = numeric_data.mean()
        # q = 0.99
        q = 0.99
        quantiles = numeric_data.quantile(q)
        normal_condition = []
        outlier_condition = []
        for col in self.numeric_columns:
            bound = 10 * quantiles[col]
            # bound = quantiles[col]
            normal_condition.append('{}<{}'.format(col, bound))
            outlier_condition.append('{}>={}'.format(col, bound))
        logger.info("outlier_condition: {}".format(outlier_condition))
        self.outliers = self.origin_df.query(' | '.join(outlier_condition))
        # tt = self.name+'_rate'
        # self.outliers[tt] = 1
        self.outliers['{}_rate'.format(self.name)] = 1
        self.origin_df = self.origin_df.query(' & '.join(normal_condition))
        self.total_rows = len(self.origin_df)
        logger.info("filtered outlier:{} rows".format(len(self.outliers)))

    def load_data(self, param):
        start_time = time.perf_counter()
        # get config parameter
        self.name = param['name']
        self.dataset_name = generate_dataset_name(param)
        header = param["header"]
        delimiter = param["delimiter"]
        filename = param["data"]
        categorical_columns = param["categorical_columns"]
        numeric_columns = param["numeric_columns"]
        logger.info("loading data:{}".format(filename))
        # load data
        if header == 1:
            df = pd.read_csv(filename, delimiter=delimiter)
        else:
            df = pd.read_csv(filename, header=None, delimiter=delimiter)

        # print("========df.std: ", df.groupby('c_nationkey')['c_acctbal'].std(ddof=0).to_dict())
        # print("========df.mean: ", df.groupby('c_nationkey')['c_acctbal'].mean().to_dict())

        self.all_columns = numeric_columns + categorical_columns
        self.origin_df = df[self.all_columns].copy(deep=True)
        self.total_rows = len(df)
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        if 'label_columns' in param and len(param['label_columns']) > 0:
            self.label_df=self.generate_label_data(param['label_columns'],param['bucket_columns'])

            # collect group count for each label group
            # self.label_group_counts = self.origin_df[self.label_column_name].value_counts().to_dict()
            self.label_group_counts = self.label_df[self.label_column_name].value_counts().to_dict()
        else:
            self.label_column_name = None
        if 'outliers' in param and param['outliers'] == 'true':
            self.filter_outlier()
        logger.info('data total rows:{}'.format(self.total_rows))
        if self.sample_for_train != 1:
            self.origin_df = self.origin_df.sample(frac=self.sample_for_train)
            self.total_rows = len(self.origin_df)
            logger.info('data total rows after sample:{}'.format(self.total_rows))
        # logger.info('label value counts:{}'.format(self.label_group_counts))
        if self.total_rows > 1000000:
            rate = 0.5
            if self.total_rows > 10000000:
                rate = 0.01
            # senate sampling
            # total_sample=int(self.total_rows * rate)
            # k = int( total_sample / len(self.label_group_counts))
            # small_groups = {g: v for g, v in self.label_group_counts.items() if k >= v}
            # big_groups = {g: v for g, v in self.label_group_counts.items() if k < v}
            # allocated=sum(small_groups.values())
            # sample_left=total_sample-allocated
            # left=sum(big_groups.values())
            # big_groups = {g: int(sample_left*v/left) for g, v in big_groups.items()}
            # allocation={**small_groups,**big_groups}
            # logger.info("sample the training data allocation:{}".format(allocation))
            # rate = {g: allocation[g] / v for g, v in self.label_group_counts.items()}
            # logger.info("sample the training data allocation rate:{}".format(rate))
            # self.origin_df = self.origin_df.groupby(self.label_column_name).apply(
            #     lambda x: x.sample(allocation[x.name]))
            # self.origin_df.reset_index(drop=True, inplace=True)

            # uniform sampling
            # self.origin_df = self.origin_df.sample(frac=rate)

            # self.total_rows = len(self.origin_df)

            logger.info("sample the training data sample rows:{}".format(self.total_rows))
        # if self.label_column_name is not None and self.label_column_name not in self.categorical_columns:
        #     self.categorical_columns.append(self.label_column_name)
        end_time = time.perf_counter()
        logger.info('load data time elapsed:{}'.format(end_time - start_time))

    def generate_label_data(self, label_columns,bucket_columns):
        label_df=self.origin_df.copy()
        label_columns_copy=label_columns.copy()
        if len(bucket_columns)>0:
            for col in bucket_columns:
                label_df[col+"_bucket"]=(label_df[col]).mod(8)
                # label_df[col+"_bucket"]=(label_df[col]).mod(4)
                col_idx=label_columns_copy.index(col)
                label_columns_copy[col_idx]=label_columns_copy[col_idx]+"_bucket"

        if len(label_columns_copy) == 1:
            self.all_categorical_columns = self.categorical_columns
            self.label_column_name = label_columns_copy[0]
            # if self.label_column_name not in self.all_categorical_columns:
            #     self.all_categorical_columns.append(self.label_column_name)
        else:
            self.label_column_name = "-".join(label_columns_copy)
            self.all_categorical_columns = self.categorical_columns #+ [self.label_column_name]
            # self.origin_df[self.label_column_name] = self.origin_df[label_columns].astype(str).sum(axis=1)
            # self.origin_df[self.label_column_name] = self.origin_df[label_columns].apply(lambda x: ''.join(x), axis=1)
            label_df[self.label_column_name] = label_df[label_columns_copy].astype(str).agg('-'.join, axis=1)
        return label_df

    def encode_categorical_data_binary(self, categorical_data):
        # binary encoding for categorical columns
        self.bce = BinaryEncoder(cols=self.categorical_columns, label=None)
        binary_encoded = self.bce.fit_transform(categorical_data)
        self.column_digits = self.bce.column_digits
        # if self.label_column_name is not None:
        #     self.label_value_mapping = self.bce.label_value_mapping
        #     self.label_mapping_out = self.bce.mapping[self.label_column_name]
        return binary_encoded

    def encode_label_binary(self, label_data):
        # binary encoding for categorical columns
        bce = BinaryEncoder(cols=[self.label_column_name], label=self.label_column_name)
        binary_encoded = bce.fit_transform(label_data)
        # self.column_digits = self.bce.column_digits
        # if self.label_column_name is not None:
        if self.delete_data == False and self.label_column_name is not None:    # 如果是删除数据，不用修改label_value_mapping
            self.label_value_mapping = bce.label_value_mapping
            self.label_mapping_out = bce.mapping[self.label_column_name]
        return binary_encoded

    def decode_categorical_data_binary(self, categorical_data):
        categorical_df = self.bce.inverse_transform(categorical_data)
        return categorical_df

    def encode_numeric_data_gaussian(self, numeric_data):
        # numeric_data = self.origin_df[self.numeric_columns]
        self.gme = GaussianEncoder(cols=self.numeric_columns, max_clusters=self.gaussian_max_clusters)
        gaussian_encoded = self.gme.fit_transform(numeric_data)
        return gaussian_encoded

    def decode_numeric_data_gaussian(self, numeric_data):
        numeric_df = self.gme.inverse_transform(numeric_data)
        return numeric_df

    def decode_samples(self, z_decoded):
        # inverse transform binary encoding categorical data
        categorical_data = z_decoded[:, self.numeric_digits:]
        numeric_data = z_decoded[:, :self.numeric_digits]
        categorical_df = None
        numeric_df = None
        if len(self.categorical_columns) > 0:
            if self.categorical_encoding == 'binary':
                categorical_df = self.decode_categorical_data_binary(categorical_data)
            else:
                categorical_df = self.decode_categorical_data_one_hot(categorical_data)

        if len(self.numeric_columns) > 0:
            if self.numeric_encoding == 'gaussian':
                numeric_df = self.decode_numeric_data_gaussian(numeric_data)
            elif self.numeric_encoding == 'stdmm':
                numeric_df = self.decode_numeric_data_stdmm(numeric_data)
            else:
                numeric_df = self.decode_numeric_data_mm(numeric_data)

        if categorical_df is None:
            sample_df = numeric_df
        elif numeric_df is None:
            sample_df = categorical_df
        else:
            sample_df = pd.concat([categorical_df, numeric_df], axis=1)
        return sample_df

    def encode_numeric_data_mm(self, numeric_data):
        self.std_scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()

        numeric_data = np.array(numeric_data)
        # numeric_data = self.std_scaler.fit_transform(numeric_data)
        numeric_data = self.mm_scaler.fit_transform(numeric_data)
        numeric_data = pd.DataFrame(numeric_data, columns=self.numeric_columns)
        return numeric_data

    def encode_numeric_data_stdmm(self, numeric_data):
        self.std_scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()

        numeric_data = np.array(numeric_data)
        numeric_data = self.std_scaler.fit_transform(numeric_data)
        numeric_data = self.mm_scaler.fit_transform(numeric_data)
        numeric_data = pd.DataFrame(numeric_data, columns=self.numeric_columns)
        return numeric_data

    def decode_numeric_data_mm(self, numeric_data):
        numeric_data = self.mm_scaler.inverse_transform(numeric_data)
        # numeric_data = self.std_scaler.inverse_transform(numeric_data)
        numeric_df = pd.DataFrame(numeric_data, columns=self.numeric_columns)
        return numeric_df

    def decode_numeric_data_stdmm(self, numeric_data):
        numeric_data = self.mm_scaler.inverse_transform(numeric_data)
        numeric_data = self.std_scaler.inverse_transform(numeric_data)
        numeric_df = pd.DataFrame(numeric_data, columns=self.numeric_columns)
        return numeric_df

    def encode_categorical_data_one_hot(self, categorical_data):
        # categorical_data = self.origin_df[all_categorical_columns]
        # one hot encoding for categorical columns
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        onehot_encoded = self.ohe.fit_transform(categorical_data).todense()

        self.column_digits = {}
        self.onehot_encoded_columns = []
        categories = self.ohe.categories_
        for idx, col in enumerate(self.all_categorical_columns):
            self.column_digits[col] = len(categories[idx])
            if self.label_column_name is not None and self.label_column_name == col:
                self.label_value_mapping = dict(enumerate(categories[idx]))
            self.onehot_encoded_columns += [col + str(i) for i in range(len(categories[idx]))]

        onehot_encoded = pd.DataFrame(onehot_encoded, columns=self.onehot_encoded_columns)
        return onehot_encoded

    def decode_categorical_data_one_hot(self, categorical_data):
        categorical_data = self.ohe.inverse_transform(categorical_data)
        categorical_df = pd.DataFrame(categorical_data, columns=self.categorical_columns)
        return categorical_df

    def change_device(self, device):
        if device != self.device:
            self.device = device
            if self.raw_data != None:
                self.raw_data = self.raw_data.to(self.device)
            if self.raw_label_data != None:
                self.raw_label_data = self.raw_label_data.to(self.device)
            print('device is changed to No.{} gpu'.format(self.device))


def generate_dataset_name(train_config):
    dataset_name = 'dataset'
    if train_config["model_type"] == "keras_vae" or train_config["model_type"] == "torch_vae":
        dataset_name = "{}_{}_{}_{}_{}".format(train_config["name"], '#'.join(train_config["categorical_columns"]),
                                               '#'.join(train_config["numeric_columns"]),
                                               train_config["categorical_encoding"],
                                               train_config["numeric_encoding"] + str(
                                                   train_config["max_clusters"]) if
                                               train_config["numeric_encoding"] == 'gaussian' else
                                               train_config["numeric_encoding"])
    elif train_config["model_type"] == "keras_cvae" or train_config["model_type"] == "torch_cvae":
        dataset_name = "{}_{}_{}_{}_{}_{}_{}".format(train_config["name"], '#'.join(train_config["categorical_columns"]),
                                                     '#'.join(train_config["numeric_columns"]),
                                                     '#'.join(train_config["label_columns"]),
                                                     train_config["categorical_encoding"],
                                                     (train_config["numeric_encoding"] + str(
                                                         train_config["max_clusters"])) if
                                                     train_config["numeric_encoding"] == 'gaussian' else
                                                     train_config["numeric_encoding"],
                                                     train_config['gpu_num'], )
    return dataset_name


def save_dataset(dataset, param, postfix=''):
    dataset_name = dataset.dataset_name  # generate_dataset_name(param)
    dataset_name += postfix
    if dataset_name.startswith('model_optimize_whole_ssales_mm'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales_mm'
    elif dataset_name.startswith('model_optimize_whole_ssales_test'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales_test'
    elif dataset_name.startswith('model_optimize_whole_ssales'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales'
    path = "./saved_datasets/{}".format(dataset_name)
    with open(path, 'wb') as file:
        pickle.dump(dataset, file, True)

    dataset.data = None
    dataset.raw_data = None
    dataset.raw_label_data = None
    dataset.origin_df = None
    light_path = "./saved_datasets/{}_light".format(dataset_name)
    with open(light_path, 'wb') as file:
        pickle.dump(dataset, file, True)


def load_dataset(train_config, postfix=''):
    start_time = time.perf_counter()
    dataset_name = generate_dataset_name(train_config)
    dataset_name += postfix
    if dataset_name.startswith('model_optimize_whole_ssales_mm'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales_mm'
    elif dataset_name.startswith('model_optimize_whole_ssales_test'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales_test'
    elif dataset_name.startswith('model_optimize_whole_ssales'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales'
    logger.info("load existing dataset:{}".format(dataset_name))
    path = "./saved_datasets/{}".format(dataset_name)
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    gpu_num = train_config['gpu_num']
    device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
    dataset.change_device(device)
    if "inc_data" in train_config and train_config['inc_train_flag'] != 'origin_train':
        dataset.inc_train_flag = train_config['inc_train_flag']
        dataset.load_incremental_data(train_config)

    end_time = time.perf_counter()
    logger.info("load dataset time elapsed:{}".format(end_time - start_time))
    return dataset

def load_light_dataset(train_config, postfix=''):
    start_time = time.perf_counter()
    dataset_name = generate_dataset_name(train_config)
    dataset_name += postfix
    if dataset_name.startswith('model_optimize_whole_ssales_mm'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales_mm'
    elif dataset_name.startswith('model_optimize_whole_ssales_test'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales_test'
    elif dataset_name.startswith('model_optimize_whole_ssales'):  # 防止文件名过长
        dataset_name = 'model_optimize_whole_ssales'
    logger.info("load existing dataset(light):{}".format(dataset_name))
    path = "./saved_datasets/{}_light".format(dataset_name)
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
        gpu_num = train_config['gpu_num']
        device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
        dataset.change_device(device)
        end_time = time.perf_counter()
        logger.info("load dataset(light) time elapsed:{}".format(end_time - start_time))
        return dataset
    return None

def exist_dataset(train_config):
    model_name = generate_dataset_name(train_config)
    if model_name.startswith('model_optimize_whole_ssales_mm'):  # 防止文件名过长
        model_name = 'model_optimize_whole_ssales_mm'
    elif model_name.startswith('model_optimize_whole_ssales_test'):  # 防止文件名过长
        model_name = 'model_optimize_whole_ssales_test'
    elif model_name.startswith('model_optimize_whole_ssales'):  # 防止文件名过长
        model_name = 'model_optimize_whole_ssales'
    path = "./saved_datasets/{}".format(model_name)
    if os.path.isfile(path):
        return True
    return False

    # def encode_categorical_data_one_hot(self):
    #     ### each categorical values domain sizes
    #     self.column_digits = {}
    #     self.column_categories_map = {}
    #     for col_name in categorical_columns:
    #         df[col_name] = df[col_name].astype("category")
    #         self.column_categories_map[col_name] = dict(enumerate(df[col_name].cat.categories))
    #         self.column_digits[col_name] = len(df[col_name].cat.categories)
    #         df[col_name] = df[col_name].cat.codes
    #         df[col_name] = df[col_name].astype("category")
    #     self.label_group_counts = df[self.label_column_name].value_counts().to_dict()
    #
    #     ### one-hot encoding for categorical columns
    #     df = pd.get_dummies(df)

    # def decode_categorical_data_one_hot(self, categorical_data):
    #     column_index = 0
    #     column_list = []
    #     ### inverse transform one-hot encoding categorical data
    #     for (col_name, col_type, col_domain_size) in self.feature_info:
    #         column_data = categorical_data[:, column_index:column_index + col_domain_size]
    #         column_index += col_domain_size
    #         if col_type == "categorical":
    #             column_data = np.argmax(column_data, axis=1).reshape(-1, 1)
    #         else:
    #             column_data = column_data
    #         column_list.append(column_data)
    #     samples = np.concatenate(column_list, axis=1)
    #     sample_df = pd.DataFrame(samples, columns=self.all_columns)
    #     for col_name in self.categorical_columns:
    #         sample_df[col_name] = sample_df[col_name].map(self.column_categories_map[col_name])
