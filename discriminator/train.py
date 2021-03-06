import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from discriminator import DomainDiscriminator
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
  tokenized_examples = tokenizer(dataset_dict['question'],
                                 dataset_dict['context'],
                                 truncation="only_second",
                                 stride=128,
                                 max_length=384,
                                 return_overflowing_tokens=True,
                                 return_offsets_mapping=True,
                                 padding='max_length')
  # Since one example might give us several features if it has a long context, we need a map from a feature to
  # its corresponding example. This key gives us just that.
  sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

  # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
  # corresponding example_id and we will store the offset mappings.
  tokenized_examples["id"] = []
  tokenized_examples["data_set_id"] = []
  for i in tqdm(range(len(tokenized_examples["input_ids"]))):
    # Grab the sequence corresponding to that example (to know what is the context and what is the question).
    sequence_ids = tokenized_examples.sequence_ids(i)
    # One example can give several spans, this is the index of the example containing this span of text.
    sample_index = sample_mapping[i]
    tokenized_examples["id"].append(dataset_dict["id"][sample_index])
    tokenized_examples["data_set_id"].append(dataset_dict["data_set_id"][sample_index])
    # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
    # position is part of the context or not.
    tokenized_examples["offset_mapping"][i] = [
      (o if sequence_ids[k] == 1 else None)
      for k, o in enumerate(tokenized_examples["offset_mapping"][i])
    ]

  return tokenized_examples

def prepare_train_data(dataset_dict, tokenizer):
  tokenized_examples = tokenizer(dataset_dict['question'],
                                 dataset_dict['context'],
                                 truncation="only_second",
                                 stride=128,
                                 max_length=384,
                                 return_overflowing_tokens=True,
                                 return_offsets_mapping=True,
                                 padding='max_length')
  sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
  offset_mapping = tokenized_examples["offset_mapping"]

  # Let's label those examples!
  tokenized_examples["start_positions"] = []
  tokenized_examples["end_positions"] = []
  tokenized_examples['id'] = []
  tokenized_examples['data_set_id'] = []
  inaccurate = 0
  for i, offsets in enumerate(tqdm(offset_mapping)):
    # We will label impossible answers with the index of the CLS token.
    input_ids = tokenized_examples["input_ids"][i]
    cls_index = input_ids.index(tokenizer.cls_token_id)

    # Grab the sequence corresponding to that example (to know what is the context and what is the question).
    sequence_ids = tokenized_examples.sequence_ids(i)

    # One example can give several spans, this is the index of the example containing this span of text.
    sample_index = sample_mapping[i]
    answer = dataset_dict['answer'][sample_index]
    # Start/end character index of the answer in the text.
    start_char = answer['answer_start'][0]
    end_char = start_char + len(answer['text'][0])
    tokenized_examples['id'].append(dataset_dict['id'][sample_index])
    tokenized_examples['data_set_id'].append(dataset_dict['data_set_id'][sample_index])

    # Start token index of the current span in the text.
    token_start_index = 0
    while sequence_ids[token_start_index] != 1:
      token_start_index += 1

    # End token index of the current span in the text.
    token_end_index = len(input_ids) - 1
    while sequence_ids[token_end_index] != 1:
      token_end_index -= 1

    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
      tokenized_examples["start_positions"].append(cls_index)
      tokenized_examples["end_positions"].append(cls_index)
    else:
      # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
      # Note: we could go after the last offset if the answer is the last word (edge case).
      while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
      tokenized_examples["start_positions"].append(token_start_index - 1)
      while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1
      tokenized_examples["end_positions"].append(token_end_index + 1)
      # assertion to check if this checks out
      context = dataset_dict['context'][sample_index]
      offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
      offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
      if context[offset_st: offset_en] != answer['text'][0]:
        inaccurate += 1

  total = len(tokenized_examples['id'])
  print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
  # print('tokenized_examples keys in prepare_train_data : ', tokenized_examples.keys())
  return tokenized_examples

def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
  # TODO: cache this if possible
  cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
  if os.path.exists(cache_path) and not args.recompute_features:
    tokenized_examples = util.load_pickle(cache_path)
  else:
    if split == 'train':
      tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
    else:
      tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
    util.save_pickle(tokenized_examples, cache_path)
  # print('tokenized_examples keys : ', tokenized_examples.keys())
  return tokenized_examples

# TODO: use a logger, use tensorboard
class Trainer():
  def __init__(self, args, log):
    self.batch_size = args.batch_size
    self.lr = args.lr
    self.discriminator_lr = args.adv_lr
    self.num_epochs = args.num_epochs
    self.device = args.device
    self.eval_every = args.eval_every
    self.path = os.path.join(args.save_dir, 'checkpoint')
    self.num_visuals = args.num_visuals
    self.save_dir = args.save_dir
    self.log = log
    self.visualize_predictions = args.visualize_predictions
    self.enable_discriminator = args.adv
    discriminator_input_size = 768
    discriminator_hidden_embed = 768
    if args.full_adv:
      discriminator_input_size = 384 * 768
      discriminator_hidden_embed = 48
    self.discriminator = DomainDiscriminator(input_size=discriminator_input_size, hidden_size=discriminator_hidden_embed)
    self.discriminator_lambda = args.adv_lambda
    self.num_adv_steps = args.adv_steps
    self.full_adv = args.full_adv
    self.enable_length_loss = args.enable_length_loss
    self.length_k = args.length_k
    self.length_lambda = args.length_lambda
    self.enable_length_bp_penalty = args.enable_length_bp_penalty
    self.length_mask = torch.ones(384, 384)
    for i in range(384):
      self.length_mask[i][i:i+self.length_k+1] = 0
    self.length_mask = torch.t(self.length_mask).to(self.device)

    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def save(self, model):
    model.save_pretrained(self.path)
    torch.save(self.discriminator.state_dict(), self.path + '/discriminator')

  def evaluate(self, model, discriminator, data_loader, data_dict, return_preds=False, split='validation'):
    device = self.device

    model.eval()
    discriminator.eval()
    pred_dict = {}
    all_start_logits = []
    all_end_logits = []
    all_dis_logits = []
    all_ground_truth_data_set_ids = []
    with torch.no_grad(), \
      tqdm(total=len(data_loader.dataset)) as progress_bar:
      for batch in data_loader:
        # Setup for forward
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        data_set_ids = batch['data_set_id'].to(device)
        batch_size = len(input_ids)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Forward
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        hidden_states = outputs.hidden_states[-1]
        _, dis_logits = self.forward_discriminator(discriminator, hidden_states, data_set_ids, full_adv=self.full_adv)

        # TODO: compute loss

        all_start_logits.append(start_logits)
        all_end_logits.append(end_logits)
        all_dis_logits.append(dis_logits)
        all_ground_truth_data_set_ids.append(data_set_ids)
        progress_bar.update(batch_size)

    # Get F1 and EM scores
    start_logits = torch.cat(all_start_logits).cpu().numpy()
    end_logits = torch.cat(all_end_logits).cpu().numpy()
    dis_logits = torch.cat(all_dis_logits).cpu().numpy()
    ground_truth_data_set_ids = torch.cat(all_ground_truth_data_set_ids).cpu().numpy()
    preds = util.postprocess_qa_predictions(data_dict,
                                            data_loader.dataset.encodings,
                                            (start_logits, end_logits))

    if split == 'validation':
      discriminator_eval_results = util.eval_discriminator(data_dict, ground_truth_data_set_ids, dis_logits)
      results = util.eval_dicts(data_dict, preds)
      results_list = [('F1', results['F1']),
                      ('EM', results['EM']),
                      ('discriminator_precision', discriminator_eval_results['precision'])]
    else:
      results_list = [('F1', -1.0),
                      ('EM', -1.0),
                      ('discriminator_precision', -1.0)]
    results = OrderedDict(results_list)
    if return_preds:
      return preds, results
    return results

  def compute_discriminator_loss(self, hidden_states, data_set_ids, full_adv):
    """
    Computes the loss for discriminator based on the hidden states of the DistillBERT model.
    Original paper implementation: https://github.com/seanie12/mrqa/blob/master/model.py
    https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html#DistilBertForQuestionAnswering
    Input: last layer hidden states of the distillBERT model, with shape [batch_size, sequence_length, hidden_dim]
    :return: loss from discriminator.
    """
    if full_adv:
      embedding = torch.flatten(hidden_states, start_dim=1)
    else:
      embedding = hidden_states[:, 0]
    log_prob = self.discriminator(embedding)
    targets = torch.ones_like(log_prob) * (1 / self.discriminator.num_classes)
    is_indomain_dataset = data_set_ids < 3
    # only back-propagate in-domain datasets
    log_prob = log_prob[is_indomain_dataset, :]
    if len(log_prob) == 0:
      return torch.tensor(0).to(self.device)
    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    return kl_criterion(log_prob, targets)

  def forward_discriminator(self, discriminator, hidden_states, data_set_ids, full_adv):
    if full_adv:
      embedding = torch.flatten(hidden_states, start_dim=1)
    else:
      embedding = hidden_states[:, 0]
    # detach the embedding making sure it's not updated from discriminator
    log_prob = discriminator(embedding.detach())
    # print('forward discriminator : ', log_prob, data_set_ids)
    criterion = nn.NLLLoss()
    loss = criterion(log_prob, data_set_ids)

    return loss, log_prob

  def train(self, model, train_dataloader, eval_dataloader, val_dict):
    device = self.device
    model.to(device)
    self.discriminator.to(device)
    qa_optim = AdamW(model.parameters(), lr=self.lr)
    dis_optim = AdamW(self.discriminator.parameters(), lr=self.discriminator_lr)

    global_idx = 0
    best_scores = {'F1': -1.0, 'EM': -1.0}
    tbx = SummaryWriter(self.save_dir)

    for epoch_num in range(self.num_epochs):
      self.log.info(f'Epoch: {epoch_num}')
      with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
        for batch in train_dataloader:
          qa_optim.zero_grad()
          dis_optim.zero_grad()
          model.train()
          self.discriminator.train()
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          start_positions = batch['start_positions'].to(device)
          end_positions = batch['end_positions'].to(device)
          data_set_ids = batch['data_set_id'].to(device)
          outputs = model(input_ids, attention_mask=attention_mask,
                          output_attentions=True,
                          output_hidden_states=True,
                          start_positions=start_positions,
                          end_positions=end_positions,
                          )
          loss = outputs[0]
          start_logits, end_logits = outputs[1], outputs[2]
          batch_size = input_ids.size(0)
          if self.enable_length_bp_penalty:
            pred_start_index = torch.argmax(start_logits, dim=1)
            pred_end_index = torch.argmax(end_logits, dim=1)
            pred_length = pred_end_index - pred_start_index + 1
            maxed_pred_length = torch.max(pred_length, torch.ones(batch_size).to(device))
            gold_length = end_positions - start_positions + 1
            maxed_gold_length = torch.max(gold_length, torch.ones(batch_size).to(device))
            weight = torch.exp(1 - maxed_gold_length / maxed_pred_length)

            filter = pred_length.cuda(device) > gold_length.cuda(device)
            weight = torch.where(filter.cuda(device), weight, torch.ones(batch_size).to(device))

            filter = pred_length.cuda(device) > 0
            weight = torch.where(filter.cuda(device), weight, torch.ones(batch_size).to(device) * 2)

            if start_positions is not None and end_positions is not None:
              if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
              if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
              ignored_index = start_logits.size(1)
              start_positions.clamp_(0, ignored_index)
              end_positions.clamp_(0, ignored_index)

              loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

              start_loss = loss_fct(start_logits, start_positions)
              end_loss = loss_fct(end_logits, end_positions)
              loss = (start_loss + end_loss) / 2 / batch_size
              loss *= weight
              loss = torch.sum(loss)
              if torch.isnan(loss):
                print('######################## loss is nan')
                print('weight : ', weight)
                print('start_loss : ', start_loss)
                print('end_loss : ', end_loss)
                print('batch_size : ', batch_size)
                print('maxed_pred_length : ', maxed_pred_length)
                print('start_positions : ', start_positions)
                print('end_positions : ', end_positions)
                print('pred_length : ', pred_length)
                print('maxed_pred_length : ', maxed_pred_length)
                print('gold_length : ', gold_length)
                print('pred_start_index : ', pred_start_index)
                print('pred_end_index : ', pred_end_index)

          scalar_length_loss = 0
          if self.enable_length_loss:
            softmax = nn.Softmax(dim=1)
            start_logits_softmax = softmax(start_logits).to(device)
            end_logits_softmax = softmax(end_logits).to(device)
            start_logits_softmax = torch.unsqueeze(start_logits_softmax, 2).to(device) # (batch, query_len, 1)
            end_logits_softmax = torch.unsqueeze(end_logits_softmax, 1).to(device) # (batch, 1, query_len)

            start_end_mul = torch.transpose(torch.matmul(start_logits_softmax, end_logits_softmax), dim0=1, dim1=1).to(device)
            length_mul = torch.matmul(start_end_mul, self.length_mask).to(device)
            diag = torch.diagonal(length_mul, dim1=1, dim2=2)
            length_loss = torch.sum(diag) / batch_size

            scalar_length_loss = (self.length_lambda * length_loss).item()
            loss += self.length_lambda * length_loss

          scalar_dis_loss_for_qa = 0
          scalar_discriminator_loss = 0
          if self.enable_discriminator:
            # hidden_states shape: [16, 384, 768]
            # [batch_size, sequence_length, hidden_size]
            hidden_states = outputs.hidden_states[-1]
            discriminator_loss_for_qa = self.discriminator_lambda * self.compute_discriminator_loss(hidden_states, data_set_ids, self.full_adv)
            scalar_dis_loss_for_qa = discriminator_loss_for_qa.item()
            loss += discriminator_loss_for_qa

            # step the qa_optim first
            loss.backward()
            qa_optim.step()
            # print('dis loss on qa : ', discriminator_loss_for_qa)
            for step in range(self.num_adv_steps):
              dis_optim.step()
              dis_optim.zero_grad()
              discriminator_loss, _ = self.forward_discriminator(self.discriminator, hidden_states, data_set_ids, self.full_adv)
              discriminator_loss.backward()
              scalar_discriminator_loss = discriminator_loss.item()
              dis_optim.step()
          else:
            loss.backward()
            qa_optim.step()
          progress_bar.update(len(input_ids))
          progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item(), dis_loss=scalar_discriminator_loss, dis_loss_on_qa=scalar_dis_loss_for_qa, length_loss=scalar_length_loss)
          tbx.add_scalar('train/NLL', loss.item(), global_idx)
          tbx.add_scalar('train/dis_loss', scalar_discriminator_loss, global_idx)
          tbx.add_scalar('train/dis_loss_on_qa', scalar_dis_loss_for_qa, global_idx)
          tbx.add_scalar('train/length_loss', scalar_length_loss, global_idx)
          # if (global_idx % self.eval_every) == 0 and global_idx > 0:
          if (global_idx % self.eval_every) == 0 and global_idx > 0:
          # TODO(lizhe): change this back
          # if (global_idx % self.eval_every) == 0:
            self.log.info(f'Evaluating at step {global_idx}...')
            preds, curr_score = self.evaluate(model, self.discriminator, eval_dataloader, val_dict, return_preds=True)
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
            self.log.info('Visualizing in TensorBoard...')
            for k, v in curr_score.items():
              tbx.add_scalar(f'val/{k}', v, global_idx)
            self.log.info(f'Eval {results_str}')
            if self.visualize_predictions:
              util.visualize(tbx,
                             pred_dict=preds,
                             gold_dict=val_dict,
                             step=global_idx,
                             split='val',
                             num_visuals=self.num_visuals)
            if curr_score['F1'] >= best_scores['F1']:
              best_scores = curr_score
              self.save(model)
          global_idx += 1
    return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name, outdomain_data_repeat):
  datasets = datasets.split(',')
  dataset_dict = None
  dataset_name = ''
  for dataset in datasets:
    dataset_name += f'_{dataset}'
    dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}', outdomain_data_repeat)
    dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
  data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
  return util.QADataset(data_encodings, train=(split_name == 'train')), dataset_dict

def main():
  # define parser and arguments
  args = get_train_test_args()

  util.set_seed(args.seed)
  model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

  if args.do_train:
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    if args.resume_training:
      checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
      model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
      model.to(args.device)
    else:
      args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(args, log)
    train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train', args.outdomain_data_repeat)
    log.info("Preparing Validation Data...")
    val_dataset, val_dict = get_dataset(args, args.eval_datasets, args.val_dir, tokenizer, 'val', 1)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=SequentialSampler(val_dataset))
    best_scores = trainer.train(model, train_loader, val_loader, val_dict)
  if args.do_eval:
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    split_name = 'test' if 'test' in args.eval_dir else 'validation'
    log = util.get_logger(args.save_dir, f'log_{split_name}')
    trainer = Trainer(args, log)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    discriminator_input_size = 768
    discriminator_hidden_embed = 768
    if args.full_adv:
      discriminator_input_size = 384 * 768
      discriminator_hidden_embed = 48
    discriminator = DomainDiscriminator(input_size=discriminator_input_size, hidden_size=discriminator_hidden_embed)
    # discriminator.load_state_dict(torch.load(checkpoint_path + '/discriminator'))
    model.to(args.device)
    discriminator.to(args.device)
    eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name, 1)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size,
                             sampler=SequentialSampler(eval_dataset))
    eval_preds, eval_scores = trainer.evaluate(model, discriminator, eval_loader,
                                               eval_dict, return_preds=True,
                                               split=split_name)
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
    log.info(f'Eval {results_str}')
    # Write submission file
    sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
      csv_writer = csv.writer(csv_fh, delimiter=',')
      csv_writer.writerow(['Id', 'Predicted'])
      for uuid in sorted(eval_preds):
        csv_writer.writerow([uuid, eval_preds[uuid]])

if __name__ == '__main__':
  main()
