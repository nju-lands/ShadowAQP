import json
import sys
import os

def get_data(config_file):
    dict = {}
    with open(config_file, 'r') as f:
        params = json.load(f)
        dict = params
    return dict

def rewrite_data(config_file, dict):
    with open(config_file, 'w') as f:
        json.dump(dict, f, indent=4)

# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['epochs'] = 250
# dict['batch_size'] = 512
# dict['latent_dim'] = 150
# dict['intermediate_dim'] = 100
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_ep250_bs512_ld_150_id_100.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['epochs'] = 200
# dict['batch_size'] = 512
# dict['latent_dim'] = 100
# dict['intermediate_dim'] = 150
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_ep200_bs512_ld_100_id_150.log 2>&1')


# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 175
# dict['latent_dim'] = 100
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep175_ld100.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 200
# dict['latent_dim'] = 100
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep200_ld100.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 250
# dict['latent_dim'] = 100
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep250_ld100.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 100
# dict['latent_dim'] = 100
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep100_ld100.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 150
# dict['latent_dim'] = 150
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep150_ld150.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 150
# dict['latent_dim'] = 50
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep150_ld50.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 200
# dict['latent_dim'] = 150
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep200_ld150.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar2.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 24
# dict['epochs'] = 100
# dict['latent_dim'] = 150
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar2.json >> ./skew_size_var/aggvar2_gaussian24_ep100_ld150.log 2>&1')


# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 36
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_gaussian34.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 37
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_gaussian34.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 38
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_gaussian34.log 2>&1')

# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 30
# dict['epochs'] = 150
# dict['latent_dim'] = 300
# dict['intermediate_dim'] = 300
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_gaussian30_ep150_ld300_id300.log 2>&1')


# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 50
# dict['epochs'] = 150
# dict['latent_dim'] = 300
# dict['intermediate_dim'] = 300
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_gaussian50_ep150_ld300_id300.log 2>&1')


# config_file = './skew_size_var/train_config/tpch_customer_aggvar15.json'
# dict = get_data(config_file)
# dict['max_clusters'] = 50
# dict['epochs'] = 150
# dict['latent_dim'] = 400
# dict['intermediate_dim'] = 400
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar15.json >> ./skew_size_var/aggvar15_gaussian50_ep150_ld400_id400.log 2>&1')


# config_file = './skew_size_var/train_config/tpch_customer_aggvar086.json'
# dict = get_data(config_file)
# dict['epochs'] = 150
# dict['latent_dim'] = 300
# dict['intermediate_dim'] = 200
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar086.json >> ./skew_size_var/aggvar086_ep150_ld300_id200.log 2>&1')


# config_file = './skew_size_var/train_config/tpch_customer_aggvar086.json'
# dict = get_data(config_file)
# dict['epochs'] = 150
# dict['latent_dim'] = 400
# dict['intermediate_dim'] = 400
# rewrite_data(config_file, dict)
# os.system('python main.py skew_size_var/query_config/tpch_aggvar086.json >> ./skew_size_var/aggvar086_ep150_ld400_id400.log 2>&1')

# tpch-cs
# config_file = 'config/query/customer_join_supplier.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.01*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cs/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cs/'+str(i)+'-2.log 2>&1')

# # tpch-cn
# config_file = 'config/query/customer_join_nation.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cn/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpch_cn/'+str(i)+'-2.log 2>&1')

# tpcds-sw
# config_file = 'config/query/ssales_join_wsales.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.01*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpcds_sw/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpcds_sw/'+str(i)+'-2.log 2>&1')

# tpcds-ss
# config_file = 'config/query/sales_join_store.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpcds_ss/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/tpcds_ss/'+str(i)+'-2.log 2>&1')

# census1
# config_file = 'config/query/census.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/PSMA/census1/'+str(i)+'.log 2>&1')
#     # os.system('python main.py ' + config_file + ' >> logs/PSMA/census1/'+str(i)+'-2.log 2>&1')

# # census2
# config_file = 'config/query/census2.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/census2-multi/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/census2-multi/'+str(i)+'-2.log 2>&1')

# flights1
# config_file = 'config/query/flights.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/flights-multi/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/flights-multi/'+str(i)+'-2.log 2>&1')

# flights2
# config_file = 'config/query/flights2.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.02*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> exp1_logs/flights2/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> exp1_logs/flights2/'+str(i)+'-2.log 2>&1')

# tpch_lpp
# config_file = 'config/query/lineitem_join_partsupp_join_parts.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lpp/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lpp/'+str(i)+'-2.log 2>&1')

# tpch_lp
# config_file = 'config/query/lineitem_join_parts.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lp/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> lihan_logs/tpch_lp/'+str(i)+'-2.log 2>&1')

# # census2
# config_file = 'config/query/census2.json'
# for i in range(1,11):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.002*i
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_select/census2-gaussian/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_select/census2-gaussian/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_select/census2-gaussian/'+str(i)+'-3.log 2>&1')

## auto update
# tpcds-sw
config_file = 'config/query/ssales_inc_join_wsales.json'
for i in range(2,11,2):
    dict = get_data(config_file)
    train_config_file = dict['train_config_files'][0]
    train_config_dict = get_data(train_config_file)
    train_config_dict['sample_rate'] = 0.01*i
    train_config_dict['inc_train_flag'] = 'old_train'
    rewrite_data(train_config_file, train_config_dict)
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/whole/'+str(i)+'-1.log 2>&1')
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/whole/'+str(i)+'-2.log 2>&1')
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/whole/'+str(i)+'-3.log 2>&1')

for i in range(2,11,2):
    dict = get_data(config_file)
    train_config_file = dict['train_config_files'][0]
    train_config_dict = get_data(train_config_file)
    train_config_dict['sample_rate'] = 0.01*i
    train_config_dict['inc_train_flag'] = 'inc_train'
    rewrite_data(train_config_file, train_config_dict)
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/inc/'+str(i)+'-1.log 2>&1')
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/inc/'+str(i)+'-2.log 2>&1')
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/inc/'+str(i)+'-3.log 2>&1')

for i in range(2,11,2):
    dict = get_data(config_file)
    train_config_file = dict['train_config_files'][0]
    train_config_dict = get_data(train_config_file)
    train_config_dict['sample_rate'] = 0.01*i
    train_config_dict['inc_train_flag'] = 'sample_train'
    rewrite_data(train_config_file, train_config_dict)
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/sample/'+str(i)+'-1.log 2>&1')
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/sample/'+str(i)+'-2.log 2>&1')
    os.system('python main.py ' + config_file + ' >> logs/auto_update/tpcds_sw/sample/'+str(i)+'-3.log 2>&1')
# tpch_cn
# config_file = 'config/query/customer_inc_join_nation.json'
# for i in range(2,11,2):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     train_config_dict['inc_train_flag'] = 'whole_train'
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/whole/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/whole/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/whole/'+str(i)+'-3.log 2>&1')

# for i in range(2,11,2):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     train_config_dict['inc_train_flag'] = 'inc_train'
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/inc/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/inc/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/inc/'+str(i)+'-3.log 2>&1')

# for i in range(2,11,2):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.001*i
#     train_config_dict['inc_train_flag'] = 'sample_train'
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/sample/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/sample/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/tpch_cn/sample/'+str(i)+'-3.log 2>&1')

# census2
# config_file = 'config/query/census2_inc.json'
# for i in range(2,11,2):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.002*i
#     train_config_dict['inc_train_flag'] = 'whole_train'
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/whole/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/whole/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/whole/'+str(i)+'-3.log 2>&1')

# for i in range(2,11,2):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.002*i
#     train_config_dict['inc_train_flag'] = 'inc_train'
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/inc/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/inc/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/inc/'+str(i)+'-3.log 2>&1')

# for i in range(2,11,2):
#     dict = get_data(config_file)
#     train_config_file = dict['train_config_files'][0]
#     train_config_dict = get_data(train_config_file)
#     train_config_dict['sample_rate'] = 0.002*i
#     train_config_dict['inc_train_flag'] = 'sample_train'
#     rewrite_data(train_config_file, train_config_dict)
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/sample/'+str(i)+'-1.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/sample/'+str(i)+'-2.log 2>&1')
#     os.system('python main.py ' + config_file + ' >> logs/auto_update/census2/sample/'+str(i)+'-3.log 2>&1')

# # census
# config_file = 'config/query/census_inc.json'
# dict = get_data(config_file)

# # "epochs": 100,
# # "batch_size": 512,
# # "latent_dim": 50,
# # "intermediate_dim": 100,

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 50
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep150_bs512_ld50_id100.log 2>&1')

# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep150_bs512_ld100_id100.log 2>&1')

# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 50
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep150_bs512_ld50_id150.log 2>&1')

# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep150_bs512_ld100_id150.log 2>&1')


# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 1024
# train_config_dict['latent_dim'] = 50
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep150_bs1024_ld50_id100.log 2>&1')


# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 100
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep100_bs512_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 100
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/auto_update/train/census_whole_ep100_bs512_ld100_id150.log 2>&1')


# flights
# config_file = 'config/query/flights.json'

# for i in range(1, 11):
#     dict = get_data(config_file)
#     dict['multi_sample_times'] = i
#     rewrite_data(config_file, dict)

#     for i in range(1000): 
#         os.system('python main.py ' + config_file)


# config_file = 'config/tpcds_whole/query/sql44.json'
# dict = get_data(config_file)

# 默认: 
# "epochs": 1,
# "batch_size": 256,
# "latent_dim": 100,
# "intermediate_dim": 100,

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 50
# train_config_dict['batch_size'] = 256
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql44/ep50_bs256_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 100
# train_config_dict['batch_size'] = 256
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql44/ep100_bs256_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 256
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql44/ep150_bs256_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 50
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql44/ep50_bs512_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 100
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql44/ep100_bs512_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 100
# train_config_dict['intermediate_dim'] = 100
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql44/ep150_bs512_ld100_id100.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 100
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 150
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql13/ep100_bs512_ld150_id150.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 512
# train_config_dict['latent_dim'] = 150
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql13/ep150_bs512_ld150_id150.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 100
# train_config_dict['batch_size'] = 128
# train_config_dict['latent_dim'] = 150
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql13/ep100_bs128_ld150_id150.log 2>&1')

# train_config_file = dict['train_config_files'][0]
# train_config_dict = get_data(train_config_file)
# train_config_dict['epochs'] = 150
# train_config_dict['batch_size'] = 128
# train_config_dict['latent_dim'] = 150
# train_config_dict['intermediate_dim'] = 150
# rewrite_data(train_config_file, train_config_dict)
# os.system('python main.py ' + config_file + ' >> logs/tpcds_whole/sql13/ep150_bs128_ld150_id150.log 2>&1')