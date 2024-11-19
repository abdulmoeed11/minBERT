import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from bert import BertModel, BertModelWithPAL
from optimizer import AdamW
from tqdm import tqdm
from optimizer_adamax import Adamax
from optimizer_SGD import SGD
from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data
from tokenizer import BertTokenizer
from evaluation import model_eval_sst, test_model_multitask


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, 1, bias=False)

    def forward(self, embeddings):
        # Apply linear transformation to the embeddings
        transformed = torch.tanh(self.W(embeddings))

        # Calculate attention weights
        attention_weights = torch.softmax(self.v(transformed), dim=1)

        # Apply attention weights to the embeddings
        attended_embeddings = torch.sum(attention_weights * embeddings, dim=1)

        return attended_embeddings


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.num_labels = config.num_labels
        print(config.num_labels)
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
                
        self.attention_layer = AttentionLayer(config.hidden_size)
        #START
                # Step 2: Add a linear layer for sentiment classification
        # self.dropout_sentiment = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(2 + 1)])
        # self.linear_sentiment = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)])
        # self.last_linear_sentiment = None

        # # Step 3: Add a linear layer for paraphrase detection
        # self.dropout_paraphrase = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(3)])
        # self.linear_paraphrase = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])

        # # Step 4: Add a linear layer for semantic textual similarity
        # # This is a regression task, so the output should be a single number
        # self.dropout_similarity = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.n_hidden_layers + 1)])
        # self.linear_similarity = nn.ModuleList([nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.n_hidden_layers)] + [nn.Linear(BERT_HIDDEN_SIZE, 1)])
        #END

        # SENTIMENT
        self.sentiment_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.sentiment_linear_out = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        # self.sentiment_layers = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     # nn.Linear(config.hidden_size, config.hidden_size),
        #     # nn.ReLU(),
        #     # nn.Linear(config.hidden_size, config.hidden_size),
        #     # nn.ReLU(),
        #     nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        # )

        # PARAPHRASE
        self.paraphrase_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.paraphrase_linear1 = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.paraphrase_linear2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.paraphrase_out = torch.nn.Linear(config.hidden_size, 2)
        # self.paraphrase_layers = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     # nn.ReLU(),
        #     # nn.Linear(config.hidden_size, config.hidden_size),
        #     # nn.ReLU(),
        #     # nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 2)
        # )

        # self.sentiment_classifier = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, 5))
        # self.paraphrase_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, 2))
        # self.similarity_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, 6))
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, attention_mask, classifier = None):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO

        result = self.bert(input_ids, attention_mask)
        attention_result = self.attention_layer(result["last_hidden_state"])
        return attention_result

        # output_dict = self.bert(input_ids, attention_mask)
        # cls = output_dict['pooler_output']
        # # logits = classifier(cls)
        # return cls


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        bert_embedding = self.forward(input_ids, attention_mask)
        logits = F.relu(self.sentiment_linear(bert_embedding))
        logits = F.relu(self.sentiment_linear1(logits))
        logits = F.relu(self.sentiment_linear2(logits))
        logits = self.sentiment_linear_out(logits)
        return logits
        # bert_embedding = self.forward(input_ids, attention_mask)
        # logits = self.sentiment_layers(bert_embedding)
        # return logits

        # bert_output = self.forward(input_ids, attention_mask, self.sentiment_classifier)
        # sentiment_logits = self.sentiment_classifier(self.dropout(bert_output))
        # return sentiment_logits

    def predict_paraphrase_train(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for predicting whether they are paraphrases.
        """
        bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        combined_bert_embeddings_1 = self.paraphrase_linear(bert_embeddings_1)
        combined_bert_embeddings_2 = self.paraphrase_linear(bert_embeddings_2)

        # Calculate absolute difference and sum of combined embeddings
        abs_diff = torch.abs(combined_bert_embeddings_1 - combined_bert_embeddings_2)
        abs_sum = torch.abs(combined_bert_embeddings_1 + combined_bert_embeddings_2)

        # Concatenate the absolute difference and sum
        concatenated_features = torch.cat((abs_diff, abs_sum), dim=1)

        # Apply linear layers to obtain logits for both "yes" and "no" predictions
        logits = F.relu(self.paraphrase_linear1(concatenated_features))
        logits = F.relu(self.paraphrase_linear2(logits))
        logits = self.paraphrase_out(logits)
        # bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        # bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        # combined_bert_embeddings_1 = self.paraphrase_layers(bert_embeddings_1)
        # combined_bert_embeddings_2 = self.paraphrase_layers(bert_embeddings_2)

        # # Calculate absolute difference and sum of combined embeddings
        # abs_diff = torch.abs(combined_bert_embeddings_1 - combined_bert_embeddings_2)
        # abs_sum = torch.abs(combined_bert_embeddings_1 + combined_bert_embeddings_2)

        # # Concatenate the absolute difference and sum
        # concatenated_features = torch.cat((abs_diff, abs_sum), dim=1)

        # # Apply linear layers to obtain logits for both "yes" and "no" predictions
        # logits = self.paraphrase_layers(concatenated_features)

        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        logits = self.predict_paraphrase_train(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
        )
        return logits.argmax(dim=-1)
        # bert_output1 = self.forward(input_ids_1, attention_mask_1, self.paraphrase_classifier)
        # bert_output2 = self.forward(input_ids_2, attention_mask_2, self.paraphrase_classifier)
        # concatenated_output = torch.cat((bert_output1, bert_output2), dim=1)
        # paraphrase_logits = self.paraphrase_classifier(self.dropout(concatenated_output))
        # return paraphrase_logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO

        bert_embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        bert_embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        diff = torch.cosine_similarity(bert_embeddings_1, bert_embeddings_2)
        return diff * 5



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(num_labels),
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'n_hidden_layers': 2
              }
    # print(len(num_labels))
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)
    lrList = [0.00001, 0.00002, 0.0001, 0.001]
    lr = args.lr
    plt.figure(figsize=(10, 6))

    # Run for the specified number of epochs
    for learnr in lrList:
        model = MultitaskBERT(config)
        model = model.to(device)
        devAcclist = []
        # optimizer = SGD(model.parameters(), lr=learnr)
        optimizer = AdamW(model.parameters(), lr=lr)
        best_dev_acc = 0
        epocList = []
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
            dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
            devAcclist.append(dev_acc)
            epocList.append(epoch)
        # Plotting
        plt.plot(epocList, devAcclist, label=learnr) 
        print("learning rate " + str(learnr)) 
        test_model(args)  
    plt.title('Dev Accuracy vs. Epochs for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Dev Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    # test_model(args)
