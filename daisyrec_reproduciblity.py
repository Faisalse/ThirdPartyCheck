import time
from logging import getLogger
from pathlib import Path
import time
import pandas as pd
from daisyRec.daisy.model.NGCFRecommender import NGCF
from daisyRec.daisy.model.LightGCNRecommender import LightGCN
from daisyRec.daisy.utils.splitter import TestSplitter
from daisyRec.daisy.utils.metrics import calc_ranking_results
from daisyRec.daisy.utils.loader import RawDataReader, Preprocessor
from daisyRec.daisy.utils.config import init_seed, init_config, init_logger
from daisyRec.daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisyRec.daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from daisyRec.daisy.utils.utils import ensure_dir, get_ur, get_history_matrix, build_candidates_set, get_inter_matrix

model_config = {
    'ngcf': NGCF,
    'lightgcn': LightGCN
}

def load_data(path):
    users = list()
    item_id = list()
    with open(path, "r") as f:
        for line in f:
            temp = line.strip()
            temp = temp.split()

            item_id.extend(temp[1:])
            users.extend( [temp[0] for i in range(len(temp[1:]))])

    df = pd.DataFrame()
    df["user"] = users
    df["item"] = item_id
    df["rating"] = [1 for i in range(len(item_id))]
    
    df["user"] = df["user"].astype("int64")
    df["item"] = df["item"].astype("int64")
    df["rating"] = df["rating"].astype("int64")
    return df



if __name__ == '__main__':

    start = time.time()
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config()

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])

    ''' init logger '''
    path = "data/"+"/"+config["dataset"]
    path =   Path(path)

    
    train_path = path / "train.txt" 
    test_path = path / "test.txt"  

    train_set = load_data(train_path)
    test_set = load_data(test_path)
    
    
    config['user_num'] = len(  train_set["user"].unique()   )
    config['item_num'] = len(  train_set["item"].unique() )
    if config["cand_num"] == "full":
        config["cand_num"]  = len(  train_set["item"].unique() )

    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger
    

    ''' get ground truth '''
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    config['train_ur'] = total_train_ur

    print("Data preprocessing is completed")

    ''' build and train model '''
    s_time = time.time()
    if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
        model = model_config[config['algo_name']](config)
        model.fit(train_set)

    elif config['algo_name'].lower() in ['multi-vae']:
        history_item_id, history_item_value, _  = get_history_matrix(train_set, config, row='user')
        config['history_item_id'], config['history_item_value'] = history_item_id, history_item_value
        model = model_config[config['algo_name']](config)
        train_dataset = AEDataset(train_set, yield_col=config['UID_NAME'])
        train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
        if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
            config['inter_matrix'] = get_inter_matrix(train_set, config)
        model = model_config[config['algo_name']](config)
        sampler = BasicNegtiveSampler(train_set, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    elif config['algo_name'].lower() in ['item2vec']:
        model = model_config[config['algo_name']](config)
        sampler = SkipGramNegativeSampler(train_set, config)
        train_samples = sampler.sampling()
        train_dataset = BasicDataset(train_samples)
        train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    else:
        raise NotImplementedError('Something went wrong when building and training...')
    elapsed_time = time.time() - s_time
    #logger.info(f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")

    ''' build candidates set '''
    logger.info('Start Calculating Metrics...')
    test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)
    end = time.time()
    ''' get predict result '''
    logger.info('==========================')
    logger.info('Generate recommend list...')
    logger.info('==========================')
    cal = end - start
    test_dataset = CandidatesDataset(test_ucands)
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    preds = model.rank(test_loader)
    
    ''' calculating KPIs '''
    logger.info('Save metric@k result to res folder...')
    path = Path("results/daisyrec/"+config["algo_name"]+"/"+config["dataset"]+"/"+"sampling_"+str(config["cand_num"]))
    
    ensure_dir(path)
    config['res_path'] = path
    print(cal)
    results = calc_ranking_results(test_ur, preds, test_u, config)

    
    results.to_csv(path / "resultFile.txt", index=False, sep = "\t")
    print(results)
    
    train_df = pd.DataFrame()
    train_df["train_time"] = [cal]
    train_df.to_csv(path / "training_time.txt", index=False, sep = "\t")

    