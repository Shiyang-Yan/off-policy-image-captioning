from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch.nn.functional as F
import time
import os
from six.moves import cPickle
import math
from seq_model_image import *
import opts
import models
from dataloader import *
import eval_utils_meshed_trigram as eval_utils
import misc.utils as utils
from rl_utils import *
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from torch.autograd import Variable
from itertools import chain
from functools import reduce
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir= 'log/')

rl_crit = utils.RewardCriterion()


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}

    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    #ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(8668, 180, 3, 0)
    models = Transformer(8667, encoder, decoder)
    # Create model
    model = models.cuda()
    lang_model = Seq2Seq().cuda()
    model.load_state_dict(torch.load('log_meshed/all2model20000.pth'))
    lang_model.load_state_dict(torch.load('language_model/langmodel06000.pth'))
    optimizer = utils.build_optimizer_adam(list(models.parameters())+ list(lang_model.parameters()), opt)
    update_lr_flag = True


    while True:

        # Update learning rate once per epoch
        if update_lr_flag:

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                #opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                #model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        # Load data from train split (0)
        start = time.time()
        data = loader.get_batch('train')
        data_time = time.time() - start
        start = time.time()

        # Unpack data
        torch.cuda.synchronize()
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['dist'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, dist_label, masks, attmasks = tmp
        labels = labels.long()

        captions = utils.decode_sequence(loader.get_vocab(), labels.view(fc_feats.size(0), -1), None)
        captions_all = []
        for index, caption in enumerate(captions):
            caption = caption.replace('<start>', '').replace(' ,', '').replace('  ', ' ')
            captions_all.append(caption)

        nd_labels = labels
        batchsize = fc_feats.size(0)
        # Forward pass and loss
        d_steps = 1
        g_steps = 1
        beta = 0.2
        #print (orch.sum(labels!=0), torch.sum(masks!=0))
        if 1:
          if 1: 
              model.train()
              optimizer.zero_grad()
              wordact, _ = model(att_feats, labels.view(batchsize, -1))
              wordact_t = wordact[:,:-1,:]
              wordact_t = wordact_t.contiguous().view(wordact_t.size(0) * wordact_t.size(1), -1)
              labels_flat = labels.view(batchsize,-1)
              wordclass_v = labels_flat[:, 1:]
              wordclass_t = wordclass_v.contiguous().view(\
               wordclass_v.size(0) * wordclass_v.size(1), -1)
              loss_xe = F.cross_entropy(wordact_t[ ...], \
               wordclass_t[...].contiguous().view(-1))

              with torch.no_grad():
                  outcap, sampled_ids, sample_logprobs, x_all_langauge, outputs, log_probs_all = lang_model.sample(labels.view(batchsize, -1).transpose(1,0), att_feats.transpose(1,0), loader.get_vocab())
              logprobs_input, _ = model(att_feats, sampled_ids.cuda().long())
              log_probs = F.log_softmax(logprobs_input[:,:,:], 2)
              sample_logprobs_true = log_probs.gather(2, sampled_ids[:,:].cuda().long().unsqueeze(2))
              with torch.no_grad():
                  reward, cider_sample, cider_greedy, caps_sample, caps = get_self_critical_reward(batchsize, lang_model, labels.view(batchsize, -1).transpose(1,0), att_feats.transpose(1,0), outcap, captions_all, loader, 180)
                  reward = torch.tensor(reward)
                  kl_div = F.kl_div(log_probs.squeeze().cuda().detach(), torch.exp(log_probs_all.transpose(1,0)).cuda().detach(), reduce= False)
                  ratio_no = sample_logprobs_true.squeeze().cpu().double()
                  ratio_de = sample_logprobs.cpu().double()
                  ratio_no_f = torch.exp(ratio_no)
                  ratio_de_f = torch.exp(ratio_de)
                  ratio = (ratio_no_f/((1-beta)*ratio_de_f+ beta*ratio_no_f))
                  ratio = torch.clamp(ratio, min = 0.96)
                  ratio_prod = ratio.prod(1)
                  reward = (torch.tensor(reward).cuda()) - 0.05 * kl_div.mean()
              loss_rl1 = rl_crit(ratio_prod.cuda().unsqueeze(1).detach()*sample_logprobs_true.squeeze()[:,:-1], sampled_ids[:,1:].cpu(), reward.float().cuda().detach())
              #writer.add_scalar('RL loss', loss_rl1 , iteration)
              #writer.add_scalar('TRIS ratio', ratio.mean(), iteration)
              #writer.add_scalar('XE_loss', loss_xe, iteration)
              #writer.add_scalar('KL_div', kl_div.mean(), iteration)
              lamb = 0.5
              train_loss = lamb * loss_rl1 + (1 - lamb)* loss_xe
              train_loss.backward()
              optimizer.step()
          
          if 1:
            if iteration % opt.print_freq == 1:
              print('Read data:', time.time() - start)
              if not sc_flag:
                  print (ratio.mean())
                  print (reward.mean())
                  print (kl_div.mean())
                  print("iter {} (epoch {}), train_loss = {:.4f}, xe_loss = {:.3f}, train_time = {:.3f}" \
                    .format(iteration, epoch, train_loss.item(), loss_xe, data_time))
              else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, np.mean(reward[:,0]), data_time, total_time))

          # Update the iteration and epoch
          iteration += 1
          if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

          # Write the training loss summary
          if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            #add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            #ss_prob_history[iteration] = model.ss_prob

        # Validate and save model 
          if (iteration % opt.save_checkpoint_every == 0):
            checkpoint_path = os.path.join(opt.checkpoint_path, 'all2model{:05d}.pth'.format(iteration))
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_path = os.path.join(opt.checkpoint_path, 'lang_model{:05d}.pth'.format(iteration))
            torch.save(lang_model.state_dict(), checkpoint_path)

            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            # Evaluate model
        #if 0:
            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            crit = utils.LanguageModelCriterion()                               
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
            # Write validation result into summary
            #add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            #if lang_stats is not None:
            #    for k,v in lang_stats.items():
            #        add_summary_value(tb_summary_writer, k, v, iteration)
            #val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Our metric is CIDEr if available, otherwise validation loss
            #if opt.language_eval == 1:
            current_score = lang_stats['CIDEr']
         #   else:
         #       current_score = - val_loss

            # Save model in checkpoint path 
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)

opt = opts.parse_opt()
opt.batch_size = 20
opt.input_att_dir =  'data/parabu_att'
opt.input_fc_dir = 'data/parabu_fc'
opt.input_json = 'data/paratalk.json'
opt.input_label_h5 = 'data/paratalk_label.h5'
opt.language_eval = 1
opt.learning_rate = 0.00015
opt.learning_rate_decay_start =0
opt.scheduled_sampling_start =0
opt.max_epochs= 20
opt.save_checkpoint_every = 1000
opt.checkpoint_path= 'log/'
opt.id = 'xe'
opt.print_freq =10
opt.model = ''
train(opt)

