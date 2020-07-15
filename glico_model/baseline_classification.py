#!/usr/bin/env python3

import csv
import time

import easyargs
import matplotlib
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn, optim
from torch.optim import lr_scheduler


matplotlib.use('Agg')
# from create_dataset_cifar100 import myDataset
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from glico_model.cifar10 import get_cifar10
from glico_model.cifar100 import get_cifar100
from glico_model.utils import get_cub_param, get_loader_with_idx, get_classifier, \
    get_test_stl_param, get_train_n_unlabled_stl_param, get_cifar10_param
from cub2011 import Cub2011

from glico_model.evaluation import fewshot_setting, accuracy
from logger import Logger


def train_linear_classifier(model, criterion, optimizer, train_labeled_dataset, num_epochs, seed, name, offset_idx=0,
                            offset_label=0, batch_size=128, fewshot=False, augment=False, autoaugment=False,
                            start_epoch=0, aug_param=None, test_data=None, cutout=False, random_erase=False, shot=None):
    if aug_param is None:
        aug_param = {'std': None, 'mean': None, 'rand_crop': 32, 'image_size': 32}
    model.train()
    sampler = None
    milestones = [60, 120, 160]
    num_epoch_repetition = 1
    if fewshot:
        num_epoch_repetition = fewshot_setting(shot, aug_param['image_size'])
        num_epochs *= num_epoch_repetition
        milestones = list(map(lambda x: num_epoch_repetition * x, milestones))
    train_data_loader = get_loader_with_idx(train_labeled_dataset, batch_size, num_workers=8,
                                            augment=augment,
                                            offset_idx=offset_idx, offset_label=offset_label, sampler=sampler,
                                            autoaugment=autoaugment, cutout=cutout, random_erase=random_erase,
                                            **aug_param)
    # train_data_loader = DataLoader(train_labeled_dataset, batch_size=128, shuffle=True, num_workers=4)
    print(f" train_len:{len(train_labeled_dataset)}")
    if aug_param['image_size'] > 32:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * num_epoch_repetition, gamma=0.9)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    use_cuda = torch.cuda.is_available()
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        end_epoch_time = time.time()
        for i, (_, inputs, targets) in enumerate(train_data_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            output = model(inputs)
            # _, preds = torch.max(output.data, 1)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum()
            acc = 100. * correct / total
        # sys.stdout.write('\r')
        # sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
        #                  % (epoch, num_epochs, i + 1,
        #                     (len(train_labeled_dataset) // batch_size) + 1, loss.item(), acc))
        # sys.stdout.flush()
        print(
                f" =>Epoch[{epoch}] acc:{acc} lr:{scheduler.get_lr()[0]}  Time: {(time.time() - end_epoch_time) / 60:8.2f}m")
        if epoch % 10 == 0 and test_data is not None:
            fewshot_acc_1_test, fewshot_acc_5_test = accuracy(model, test_data, batch_size=batch_size,
                                                              aug_param=aug_param)
            print(f"\n=> Mid accuracy@1: {fewshot_acc_1_test} | accuracy@5: {fewshot_acc_5_test}\n")
            model.train()
        # if epoch % 5 == 0:
        # 	checkpoint(acc, epoch, model, seed, name)
        scheduler.step()


# save_checkpoint(f'baseline/{name}', epoch, model)


@easyargs
def run_eval_generic(seed=17, epochs=200, d="", fewshot=False, augment=False, autoaugment=False, data="", shot=0,
                     unlabeled_shot=None,
                     resume=False, pretrained=False, batch_size=128, cutout=False, random_erase=False):
    global classifier
    dir = "baselines_aug"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    name = f"{data}_{d}_baseline_"
    if not shot is None:
        name = name + f"shot{shot}"
    if not autoaugment is None:
        name = name + "_aug"

    logger = Logger(os.path.join(f"{dir}/", f'{d}_log.txt'), title=name)
    logger.set_names(["valid_acc@1", "valid_acc@5", "test_acc@1", "test_acc@5"])
    data_dir = '../../data'
    cifar_dir_cs = '/cs/dataset/CIFAR/'
    aug_param = aug_param_test = None

    if data == "cifar":
        classes = 100
        # batch_size = 128
        lr = 0.1
        WD = 5e-4
        if not fewshot:
            train_labeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
            train_unlabeled_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
            test_data = train_unlabeled_dataset
        else:
            print("=> Fewshot")
            # train_labeled_dataset, train_unlabeled_dataset = get_cifar100_small(cifar_dir_small, shot)
            train_labeled_dataset, train_unlabeled_dataset, _, test_data = get_cifar100(cifar_dir_cs, n_labeled=shot,
                                                                                        n_unlabled=unlabeled_shot)
        # test_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
        print(f"train_labeled_dataset size:{len(train_labeled_dataset)}")
        print(f"test_data size:{len(test_data)}")

    if data == "cifar-10":
        classes = 10
        lr = 0.1
        WD = 5e-4
        aug_param = get_cifar10_param()
        if not fewshot:
            train_labeled_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
            train_unlabeled_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
            test_data = train_unlabeled_dataset
        else:
            print("=> Fewshot")
            train_labeled_dataset, train_unlabeled_dataset, _, test_data = get_cifar10(cifar_dir_cs, n_labeled=shot,
                                                                                       n_unlabled=unlabeled_shot)
    print(f"train_labeled_dataset size:{len(train_labeled_dataset)}")
    print(f"test_data size:{len(test_data)}")

    if data == "cub":
        print("CUB-200")
        classes = 200
        # batch_size = 16
        lr = 0.001
        WD = 1e-5
        aug_param = aug_param_test = get_cub_param()
        split_file = None
        if fewshot:
            samples_per_class = int(shot)
            split_file = 'train_test_split_{}.txt'.format(samples_per_class)
            train_labeled_dataset = Cub2011(root=f"../../data/{data}", train=True, split_file=split_file)
        test_data = Cub2011(root=f"../../data/{data}", train=False)
        print(f"train_labeled_dataset size:{len(train_labeled_dataset)},{train_labeled_dataset.data.shape}")

        print(f"test_data size:{len(test_data)},{test_data.data.shape}")
    if data == "stl":
        print("STL-10")
        classes = 10
        WD = 4e-4
        # batch_size = 128
        lr = 2e-3
        aug_param = get_train_n_unlabled_stl_param()
        aug_param_test = get_test_stl_param()
        train_labeled_dataset = torchvision.datasets.STL10(root=f"../../data/{data}", split='train', download=True)
        # train_unlabeled_dataset = torchvision.datasets.STL10(root=f"../../data/{data}", split='unlabeled', download=True)
        test_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='test', download=True)
        print(f"train_labeled_dataset size:{len(train_labeled_dataset)},{train_labeled_dataset.data.shape}")
        print(f"test_data size:{len(test_data)},{test_data.data.shape}")
    scores_test_acc1_fewshot = dict()
    scores_test_acc5_fewshot = dict()

    # offset_idx = len(train_labeled_dataset)
    print(f"{data} num classes: {classes}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_epoch = 0
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./checkpoint/{name}_{seed}_ckpt.t7')
        classifier = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
        print(f"=> Model loaded start_epoch{start_epoch}, acc={best_acc}")

    else:

        classifier = get_classifier(classes, d, pretrained)
    criterion = nn.CrossEntropyLoss().cuda()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        classifier = nn.DataParallel(classifier).cuda()
    else:
        classifier = classifier.cuda()
    optimizer = optim.SGD(classifier.parameters(), lr, momentum=0.9, weight_decay=WD, nesterov=True)
    cudnn.benchmark = True
    print(' => Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))
    print(f"=> {d}  Training model")
    print(f"=> Training Epochs = {str(epochs)}")
    print(f"=> Initial Learning Rate = {str(lr)}")

    # criterion = maybe_cuda(criterion)
    # optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    train_linear_classifier(classifier, criterion, optimizer, seed=seed, name=name,
                            train_labeled_dataset=train_labeled_dataset,
                            batch_size=batch_size,
                            num_epochs=epochs,
                            fewshot=fewshot,
                            augment=augment, autoaugment=autoaugment, aug_param=aug_param, test_data=test_data,
                            start_epoch=start_epoch, cutout=cutout, random_erase=random_erase, shot=shot)

    acc_1_valid, acc_5_valid = accuracy(classifier, train_labeled_dataset, batch_size=batch_size, aug_param=aug_param)
    fewshot_acc_1_test, fewshot_acc_5_test = accuracy(classifier, test_data, batch_size=batch_size,
                                                      aug_param=aug_param_test)

    print('fewshot_acc_1_valid accuracy@1', acc_1_valid)
    print('fewshot_acc_5_valid accuracy@1', acc_5_valid)

    print('fewshot_acc_1_test accuracy@1', fewshot_acc_1_test)
    print('fewshot_acc_5_test accuracy@5', fewshot_acc_5_test)
    logger.append([acc_1_valid, acc_5_valid, fewshot_acc_1_test, fewshot_acc_5_test])
    scores_test_acc1_fewshot[seed] = fewshot_acc_1_test
    scores_test_acc5_fewshot[seed] = fewshot_acc_5_test

    w = csv.writer(open(f"baseline_shot_{shot}_acc1.csv", "w+"))
    for key, val in scores_test_acc1_fewshot.items():
        w.writerow([key, val])
    w = csv.writer(open(f"baseline_shot_{shot}_acc5.csv", "w+"))
    for key, val in scores_test_acc5_fewshot.items():
        w.writerow([key, val])
    logger.close()


def checkpoint(acc, epoch, net, seed, name):
    # Save checkpoint.
    print('Saving..')
    state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
            }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/{name}_{seed}_ckpt.t7')


if __name__ == '__main__':
    run_eval_generic()
    print("Done!")
