import argparse
import os
import random
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import models
import utils

from use_meta_dataset import make_md


def main(config):
    svname = args.name
    if svname is None:
        svname = 'pretrain-multi'
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    def make_dataset(name):
        dataset = make_md([name],
            'batch', split='train', image_size=126, batch_size=256)
        return dataset

    ds_names = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', \
            'quickdraw', 'fungi', 'vgg_flower']
    datasets = []
    for name in ds_names:
        datasets.append(make_dataset(name))
    iters = []
    for d in datasets:
        iters.append(d.make_one_shot_iterator().get_next())

    to_torch_labels = lambda a: torch.from_numpy(a).long()

    to_pil = transforms.ToPILImage()
    augmentation = transforms.Compose([
        transforms.Resize(146),
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    ########

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

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

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        n_batch = 915547 // 256
        with tf.Session() as sess:
            for i_batch in tqdm(range(n_batch)):
                if random.randint(0, 1) == 0:
                    ds_id = 0
                else:
                    ds_id = random.randint(1, len(datasets) - 1)

                next_element = iters[ds_id]
                e, cfr_id = sess.run(next_element)

                data_, label = e[0], to_torch_labels(e[1])
                data_ = ((data_ + 1.0) * 0.5 * 255).astype('uint8')
                data = torch.zeros(256, 3, 128, 128).float()
                for i in range(len(data_)):
                    x = data_[i]
                    x = to_pil(x)
                    x = augmentation(x)
                    data[i] = x

                data = data.cuda()
                label = label.cuda()

                logits = model(data, cfr_id=ds_id)
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)

                logits = None; loss = None

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

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
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    utils.set_gpu(args.gpu)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['_parallel'] = True

    main(config)

