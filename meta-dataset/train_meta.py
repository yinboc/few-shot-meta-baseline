import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorboardX import SummaryWriter

import models
import utils
from use_meta_dataset import make_md
from torchvision import transforms


def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta'
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    if args.dataset == 'all':
        train_lst = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd',
                'quickdraw', 'fungi', 'vgg_flower']
        eval_lst = ['ilsvrc_2012']
    else:
        train_lst = [args.dataset]
        eval_lst = [args.dataset]

    if config.get('no_train') == True:
        train_iter = None
    else:
        trainset = make_md(train_lst, 'episodic', split='train', image_size=126)
        train_iter = trainset.make_one_shot_iterator().get_next()

    if config.get('no_val') == True:
        val_iter = None
    else:
        valset = make_md(eval_lst, 'episodic', split='val', image_size=126)
        val_iter = valset.make_one_shot_iterator().get_next()

    testset = make_md(eval_lst, 'episodic', split='test', image_size=126)
    test_iter = testset.make_one_shot_iterator().get_next()

    sess = tf.Session()

    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    def process_data(e):
        e = list(e[0])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(146),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        for ii in [0, 3]:
            e[ii] = ((e[ii] + 1.0) * 0.5 * 255).astype('uint8')
            tmp = torch.zeros(len(e[ii]), 3, 128, 128).float()
            for i in range(len(e[ii])):
                tmp[i] = transform(e[ii][i])
            e[ii] = tmp.cuda()

        e[1] = torch.from_numpy(e[1]).long().cuda()
        e[4] = torch.from_numpy(e[4]).long().cuda()

        return e

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if config.get('no_train') == True:
            pass
        else:
            for i_ep in tqdm(range(config['n_train'])):

                e = process_data(sess.run(train_iter))
                loss, acc = model(e[0], e[1], e[3], e[4])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)

                loss = None 

        # eval
        model.eval()

        for name, ds_iter, name_l, name_a in [
                ('tval', val_iter, 'tvl', 'tva'),
                ('val', test_iter, 'vl', 'va')]:
            if config.get('no_val') == True and name == 'tval':
                continue

            for i_ep in tqdm(range(config['n_eval'])):

                e = process_data(sess.run(ds_iter))

                with torch.no_grad():
                    loss, acc = model(e[0], e[1], e[3], e[4])
                
                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        _sig = 0

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default='all')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

