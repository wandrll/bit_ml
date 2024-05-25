import torch
import yaml
from models import trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger
from models.seq2seq_transformer import Seq2SeqTransformer

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("configs/data_config.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", 'r'),   Loader=yaml.Loader)

    # TODO: Инициализируйте модель Seq2SeqTransformer
    model = Seq2SeqTransformer(device=DEVICE,
                               emb_size=model_config['embedding_size'],
                               vocab_size=model_config['max_vocab_size'],
                               max_seq_len=model_config['max_seq_len'],
                               target_tokenizer=dm.target_tokenizer,
                               nhead=model_config['nhead'],
                               step_size=model_config['sched_step_size'],
                               gamma=model_config['sched_gamma'],
                               dim_feedforward=model_config['dim_feedforward'])

    logger = TXTLogger('training_log')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)




