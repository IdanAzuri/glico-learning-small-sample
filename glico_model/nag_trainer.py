from __future__ import absolute_import, division, print_function, unicode_literals

import socket
import time

import matplotlib
import os
import shutil
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torchvision.models import resnet50

matplotlib.use('Agg')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from glico_model.model import Classifier
from glico_model.interpolate import save_inter_imgs
from glico_model import model
from glico_model.utils import *
from logger import Logger, savefig


class NAG():
    def __init__(self, nag_params, image_params, rn, resume, num_classes, resolution, datasets=None, is_ft=False):
        self.num_classes = num_classes
        self.noise2niose = False
        if self.noise2niose:
            self.noise_augment = AugmentGaussian()
        self.consecutive_loss = 0
        self.best_loss = 1000
        self.vis_n = 64
        self.code_size = 100
        self.fixed_noise = maybe_cuda(torch.FloatTensor(self.vis_n, nag_params.nz).normal_(0, 0.04))
        self.fixed_code = torch.cuda.FloatTensor(self.vis_n, self.code_size).normal_(0, 0.15)
        # normed_fixed_code = fixed_code.norm(2, 1).detach().unsqueeze(1).expand_as(fixed_code)
        # self.fixed_code = fixed_code.div(normed_fixed_code)
        if nag_params.data_name == 'cub':
            self.aug_param = get_cub_param()
        if nag_params.data_name == 'cifar-100':
            self.aug_param = get_cifar100_param()
        if nag_params.data_name == 'cifar-10':
            self.aug_param = get_cifar10_param()
        if nag_params.data_name == 'imagenet':
            self.aug_param = get_imagenet_param()
        if nag_params.data_name == 'stl':
            self.aug_param = get_train_stl_param()
        self.rn = rn
        self.epoch = 0
        offset_labels = 0
        data_loader_for_init = get_loader_with_idx(datasets[0], batch_size=1, shuffle=False, **self.aug_param)
        offset_idx = len(datasets[0])
        data_loader_for_init_fewshot = []
        # self.is_cifar = "cifar" not in str.lower(str(datasets[0].__class__))
        # if self.is_cifar:
        # 	print(f"=> Offset lables! {str.lower(str(datasets[0].__class__))}")
        # 	offset_labels = max(datasets[0].targets) + 1
        print(f"[DEBUG] dataset name: {str.lower(str(datasets[0].__class__))},offset_labels={offset_labels}")
        if len(datasets[1]) > 0:
            print(f"offset_idx1:{offset_idx}")
            data_loader_for_init_fewshot = get_loader_with_idx(datasets[1], batch_size=1, shuffle=False, augment=False,
                                                               offset_idx=offset_idx,
                                                               offset_label=offset_labels, **self.aug_param)
        data_loader_for_init_rest = []
        if len(datasets[2]) > 0:
            offset_idx = offset_idx + len(datasets[1])
            print(f"offset_idx2:{offset_idx}")
            data_loader_for_init_rest = get_loader_with_idx(datasets[2], batch_size=1, shuffle=False, augment=False,
                                                            offset_idx=offset_idx,
                                                            offset_label=offset_labels, **self.aug_param)
        self.netZ = model._netZ(nag_params.nz, image_params.n, num_classes,
                                [data_loader_for_init, data_loader_for_init_fewshot, data_loader_for_init_rest])
        if nag_params.data_name == 'cifar-100' or nag_params.data_name == 'cifar-10':
            self.netG = model._netG(nag_params.nz, image_params.sz[0], image_params.nc, nag_params.noise_proj,
                                    nag_params.noise_proj)
        elif nag_params.data_name == 'stl':
            self.netG = model.DCGAN_G_small(nag_params.nz, image_params.sz[0], image_params.nc, nag_params.noise_proj,
                                            nag_params.noise_proj)
        elif nag_params.data_name == 'cub':
            self.netG = model.DCGAN_G(nag_params.nz, image_params.sz[0], image_params.nc, nag_params.noise_proj,
                                      nag_params.noise_proj)

        self.num_gpu = torch.cuda.device_count()
        self.nag_params = nag_params
        self.image_params = image_params
        # if "samba" not in socket.gethostname():
        # 	self.tensorboard = TensorBoard(f'runs/{self.rn}')
        self.logger = Logger(os.path.join(f'runs/', f'{self.rn}_log.txt'), title=self.rn)
        # self.logger_wighets = Logger(os.path.join(f'runs/', f'{self.rn}_log_W.txt'), title=self.rn)
        # self.logger_wighets.set_names(['train_mean','train_max','train_min','transd_mean','transd_max','transd_min'])
        self.logger.set_names(['loss', 'd_loss', 'rec_loss'])
        self.is_ft = is_ft
        self.milestones = [60, 120, 160]
        if self.nag_params.is_classifier:
            if self.nag_params.is_pixel:
                if self.nag_params.disc_net == "vgg":
                    self.classifier = vgg19_bn(num_classes=self.num_classes)
                elif self.nag_params.disc_net == "wideresnet":
                    self.classifier = WideResNet(depth=16, num_classes=self.num_classes, widen_factor=8, dropRate=0.3)
                elif self.nag_params.disc_net == "resnet":
                    self.classifier = resnet50(pretrained=False, num_classes=self.num_classes)
                elif self.nag_params.disc_net == "conv":
                    self.classifier = cnn(num_classes=self.num_classes, im_size=resolution)  # Pixel space

            else:
                self.classifier = Classifier(self.num_classes, dim=nag_params.nz)

        if resume:
            self.load_models()
        else:
            self.init_models_weights()
        self.old_embedding_weights = self.netZ.emb.weight.data.detach().cpu().numpy()
        if self.nag_params.loss == "cosine":
            self.d_criterion = nn.CosineEmbeddingLoss()
        else:
            self.d_criterion = nn.CrossEntropyLoss()

        if self.num_gpu > 1:
            print(f"=>Using {self.num_gpu} GPUs!")
            self.netG = nn.DataParallel(self.netG.cuda(), device_ids=list(range(self.num_gpu)))
            # self.netZ = nn.DataParallel(self.netZ.cuda(), device_ids=list(range(self.num_gpu)))
            self.d_criterion = nn.DataParallel(self.d_criterion.cuda(), device_ids=list(range(self.num_gpu)))
            if self.nag_params.is_classifier:
                self.classifier = nn.DataParallel(self.classifier.cuda(), device_ids=list(range(self.num_gpu)))
            self.netZ.cuda()
            cudnn.benchmark = True
        else:
            maybe_cuda(self.netZ)
            maybe_cuda(self.netG)
            if self.nag_params.is_classifier:
                maybe_cuda(self.classifier)
                maybe_cuda(self.d_criterion)

    def load_models(self):
        try:
            print(f"=> Loading model from {self.rn}")
            _, self.netG = load_saved_model(f'runs/nets_{self.rn}/netG_nag', self.netG)
            self.epoch, self.netZ = load_saved_model(f'runs/nets_{self.rn}/netZ_nag', self.netZ)
            if self.nag_params.is_classifier:
                _, self.classifier = load_saved_model(f'runs/nets_{self.rn}/netD_nag', self.classifier)

            if self.epoch > 0:
                print(f"=> Loaded successfully! epoch:{self.epoch}")
                if self.is_ft:
                    self.netZ.ft_maps_from_loader()
            else:
                print("=> No checkpoint to resume")
                self.init_models_weights()

        except Exception as e:
            print(f"=>Failed resume job!\n {e}")
            self.init_models_weights()

    def init_models_weights(self):
        if self.nag_params.z_init == "rndm":
            print(f"init rndm")
            self.netZ.apply(model.weights_init)
            self.netZ.set_maps_from_loader()
        if self.nag_params.z_init == "cube":
            print(f"init cube")
            self.netZ.apply(model.weights_init)
            self.netZ.cube_init()
        if self.nag_params.z_init == "resnet":
            print(f"init resnet")
            self.netZ.apply(model.weights_init)
            self.netZ.resnet_init()
        self.netG.apply(model.weights_init)
        if self.nag_params.is_classifier:
            self.classifier.apply(model.weights_init)
        del self.netZ.data_loader

        if self.nag_params.shot > 0:
            for k in self.netZ.label2idx.keys():
                assert (len(self.netZ.label2idx[k])) <= self.nag_params.shot, f"len: {len(self.netZ.label2idx[k])}"

    def check_consecutive(self, batch_size):
        if self.consecutive_loss > 10:
            self.consecutive_loss = 0
            print("=>Update batch_size")
            # lr = lr * opt_params.decay_rate
            batch_size = min(batch_size * 2, 1024)
        return batch_size

    def train_long(self, train_data, unlabeled_train_data, test_data_transductive, opt_params, resolution,
                   vis_epochs=50, save_ckpt=5):
        global epoch_loss, d_loss, epoch_loss_few_shot, d_loss_few_shot
        unlabeled_data_loader = optimizerD = scheduler = transductive_data_loader = None
        epoch_loss_few_shot = 0
        d_loss_few_shot = 0
        epoch = self.epoch
        batch_size = opt_params.batch_size
        gamma = opt_params.gamma
        # loss citertion
        self.dist = distance_metric(resolution, self.image_params.nc, self.nag_params.force_l2)
        self.dist = nn.DataParallel(self.dist, device_ids=list(range(self.num_gpu)))
        train_data_loader = get_loader_with_idx(train_data, batch_size, **self.aug_param)
        if len(unlabeled_train_data) > 0:
            unlabeled_data_loader = get_loader_with_idx(unlabeled_train_data, batch_size,
                                                        offset_idx=len(train_data),
                                                        offset_label=0,
                                                        **self.aug_param)  # TODO: get_label_max(train_data) that for CIFAR ONLY!
        is_transductive = len(test_data_transductive) > 0
        if is_transductive:
            print("transductive training")
            transductive_data_loader = get_loader_with_idx(test_data_transductive, batch_size,
                                                           offset_idx=len(unlabeled_train_data) + len(train_data),
                                                           offset_label=0,
                                                           **self.aug_param)  # TODO only for CIFAR len(train_data.classes)
        while epoch < opt_params.epochs:
            decay_steps = epoch // opt_params.decay_epochs
            lr = opt_params.lr * opt_params.decay_rate ** decay_steps
            print(f"=>epoch:{epoch}/{opt_params.epochs}  resolution:{resolution} batch:{batch_size}, lr{lr:e}")
            end_epoch_time = time.time()

            # batch_size = self.check_consecutive(batch_size)
            optimizerG = optim.Adam(self.netG.parameters(), lr=lr * opt_params.factor, betas=(0.5, 0.999))
            optimizerZ = optim.Adam(self.netZ.parameters(), lr=lr, betas=(0.5, 0.999))
            if self.nag_params.is_classifier:
                # optimizerD = optim.Adam(
                # 	[{'params': self.classifier.parameters()}, {'params': self.netZ.parameters(), 'lr': 1e-3}],
                # 	lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-4)
                optimizerD = optim.SGD(self.netZ.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                optimizerD = optim.SGD([
                        {'params': self.classifier.parameters()},
                        {'params': self.netZ.parameters(), 'lr': 1e-3}],
                        0.1, momentum=0.9, weight_decay=1e-4)
                scheduler = lr_scheduler.MultiStepLR(optimizerD, milestones=self.milestones, gamma=0.2)
            # scheduler = lr_scheduler.CosineAnnealingLr(optimizerD, T_max=10,eta_min=0)
            epoch_loss, d_loss = self.train_epoch(train_data_loader, epoch=epoch, optim_Z=optimizerZ,
                                                  optim_G=optimizerG, optim_D=optimizerD, scheduler=scheduler,
                                                  gamma=gamma)
            if unlabeled_data_loader is not None:
                _, _ = self.train_epoch(unlabeled_data_loader, epoch=epoch,
                                        optim_Z=optimizerZ, optim_G=optimizerG,
                                        optim_D=optimizerD, is_transductive=True, scheduler=scheduler, gamma=gamma)
            if is_transductive:
                epoch_loss_transductive, _ = self.train_epoch(transductive_data_loader, epoch=epoch,
                                                              optim_Z=optimizerZ, optim_G=optimizerG,
                                                              optim_D=optimizerD,
                                                              is_transductive=True,
                                                              scheduler=scheduler, gamma=gamma)
            # self.tensorboard._add_summary(epoch, {"rec_loss_transductive": epoch_loss_transductive,
            #                                       "d_loss_transductive": d_loss_transductive})
            self.logger.append([epoch_loss, d_loss, epoch_loss - d_loss])
            print(f"=>NAG Epoch: {epoch} Error: {epoch_loss}, Time: {(time.time() - end_epoch_time) / 60:8.2f}m")
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                if epoch % save_ckpt == 0:
                    self.save(epoch)
                self.consecutive_loss = 0
            else:
                self.consecutive_loss += 1
                print(f"=>consecutive_loss: {self.consecutive_loss}")

            if epoch % vis_epochs == 0:
                try:
                    self.visualize_no_module(epoch, self.aug_param, train_data.data)
                except Exception as e:
                    print(e)
            epoch += 1
        self.logger.close()
        self.logger.plot()
        savefig(self.rn + "plt.eps")

    def check_consecutive(self, batch_size):
        if self.consecutive_loss > 10:
            self.consecutive_loss = 0
            print("=>Update batch_size")
            # lr = lr * opt_params.decay_rate
            batch_size = min(batch_size * 2, 1024)
        return batch_size

    def save(self, epoch):
        try:
            save_checkpoint(f'runs/nets_{self.rn}/netZ_nag', epoch, self.netZ,
                            [self.netZ.label2idx, self.netZ.idx2label])
            save_checkpoint(f'runs/nets_{self.rn}/netG_nag', epoch, self.netG)
            if self.nag_params.is_classifier:
                save_checkpoint(f'runs/nets_{self.rn}/netD_nag', epoch, self.classifier)
        except:
            print("failed to save!")

    def train_epoch(self, data_loader, epoch, optim_Z, optim_G, optim_D, scheduler, gamma, is_transductive=False):
        '''

        :param is_transductive: train without labels
        :return:
        '''
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.5f')
        d_losses = AverageMeter('d_Loss', ':6.5f')
        progress = ProgressMeter(len(data_loader), batch_time, data_time, losses, d_losses,
                                 prefix="Epoch: [{}]".format(epoch))
        # Train
        epoch_loss = 0
        end = time.time()
        for i, batch in enumerate(data_loader):
            data_time.update(time.time() - end)
            d_loss = 0
            idx = maybe_cuda(batch[0])
            if self.noise2niose:
                images = self.noise_augment.add_gaussian(maybe_cuda(batch[1]))
            else:
                images = maybe_cuda(batch[1])
            labels = maybe_cuda(batch[2])
            self.batch_size = len(idx)  # .shape[0

            code = torch.cuda.FloatTensor(self.batch_size, self.code_size).normal_(0, 0.15)
            # normed_code = code.norm(2, 1).detach().unsqueeze(1).expand_as(code)
            # code = code.div(normed_code)

            zi = self.netZ(idx)
            if self.noise2niose:
                Ii = self.netG(self.noise_augment.add_gaussian(zi), code)
            else:
                Ii = self.netG(zi, code)

            if self.nag_params.is_classifier and not is_transductive:
                d_loss = self.get_classifier_loss(labels, zi, code)
                d_losses.update(d_loss, self.batch_size)
            epoch_loss = self.make_step(Ii, images, losses, optim_G, optim_Z, optim_D, gamma=gamma, d_loss=d_loss)
            if self.nag_params.is_classifier:
                scheduler.step()
            batch_time.update(time.time() - end)
            if i % 50 == 0:
                progress.print(i)

            self.netZ.get_norm()

        return epoch_loss, d_losses.avg

    def make_step(self, Ii, images, losses, optimizerG, optimizerZ, optimizerD, gamma, d_loss=0):
        rec_loss = self.dist(2 * Ii - 1, 2 * images - 1)
        rec_loss = rec_loss.mean() + gamma * d_loss  # + slerp_loss
        # Backward pass and optimization step
        optimizerZ.zero_grad()
        optimizerG.zero_grad()
        if self.nag_params.is_classifier:
            optimizerD.zero_grad()
        rec_loss.backward()
        optimizerZ.step()
        optimizerG.step()
        if self.nag_params.is_classifier:
            optimizerD.step()
        losses.update(rec_loss.item(), self.batch_size)
        return losses.avg

    def get_classifier_loss(self, labels, zi, code):
        if self.nag_params.loss == "ce_smooth":  # cross entrop smoothing
            onehot_labels = one_hot(labels, self.num_classes)
            label_smoothing = 0.1
            labels = (onehot_labels * (1 - label_smoothing) + 0.5 * label_smoothing)
        if self.nag_params.is_pixel:
            gen = self.netG(zi, code)
            outputs = self.classifier(gen)
        else:
            outputs = self.classifier(zi)
        if self.nag_params.loss == "cosine":
            self.y_target = torch.ones(len(labels)).cuda()
            onehot_labels = one_hot(labels, self.num_classes)
            zd_loss = self.d_criterion(outputs, onehot_labels.cuda(), self.y_target)
        else:
            d_loss = self.d_criterion(outputs, labels)
        return d_loss.mean()

    def visualize_no_module(self, epoch, aug_param, imgs):
        print("Show images...")
        # Reconstructed
        image_size = aug_param['rand_crop']
        idx = maybe_cuda(torch.from_numpy(np.arange(self.vis_n)))
        # targets=validate_loader_consistency(self.netZ,idx.detach().cpu())
        if self.noise2niose:
            Irec_noisy = self.netG(self.noise_augment.add_validation_noise(self.netZ(idx)), self.fixed_code)[
                         :self.vis_n]
            # save_image_grid(Irec.data, f'runs/ims_{self.rn}/reconstructions{image_size}_{epoch}_grid.png')
            vutils.save_image(Irec_noisy.data,
                              f'runs/ims_{self.rn}/reconstructions_niosy_{image_size}__{epoch}_vutils.png',
                              normalize=False)
        Irec = self.netG(self.netZ(idx), self.fixed_code)[:self.vis_n]
        # save_image_grid(Irec.data, f'runs/ims_{self.rn}/reconstructions{image_size}_{epoch}_grid.png')
        vutils.save_image(Irec.data, f'runs/ims_{self.rn}/reconstructions_{image_size}__{epoch}_vutils.png',
                          normalize=False)
        # Generated images + noise
        Igen_noise = self.netG(self.fixed_noise + self.netZ(idx), self.fixed_code)
        vutils.save_image(Igen_noise.data, f'runs/ims_{self.rn}/generation_noise_{image_size}__{epoch}_vutils.png',
                          normalize=False)
        # Generated images
        Igen = self.netG(self.fixed_noise, self.fixed_code)
        vutils.save_image(Igen.data, f'runs/ims_{self.rn}/generation_{image_size}__{epoch}_vutils.png', normalize=False)
        # Gaussian
        z = sample_gaussian(self.netZ.emb.weight.clone().cpu(), self.vis_n)
        Igauss = self.netG(z, self.fixed_code)
        vutils.save_image(Igauss.data, f'runs/ims_{self.rn}/gauss_{image_size}__{epoch}_vutils.png', normalize=False)
        save_inter_imgs(self.netZ, self.netG, self.rn)


# try:
# 	plot_kernels(self.netG.conv1,rn=self.rn,epoch=epoch)
# except Exception as e:
# 	print(e)
# 	print(traceback.format_exc())


# try:
# 	tsne_vis(self.netZ,self.rn,aug_param['rand_crop'],real_imgs=imgs)
# except:
# 	pass


class NAGTrainer():
    def __init__(self, dataset, nag_params, rn, resume=False, num_classes=10):
        # ims_np=np.array(ims_np)

        global resolution
        self.data_tup = dataset
        # self.is_cifar = "cifar" in str.lower(str(dataset[0].__class__))
        if 'cifar' in nag_params.data_name:
            resolution = 32
        elif nag_params.data_name == 'imagenet':
            resolution = 84
        elif nag_params.data_name == 'cub':
            resolution = 256
        elif nag_params.data_name == 'stl':
            resolution = 96
        print(f"[DEBUG] data name={nag_params.data_name}, data res = {resolution}")
        self.sz = (resolution, resolution)  # ims_np[0][0].shape[1:2]
        self.rn = rn
        self.create_dirs(resume)
        self.nc = 3
        self.n = sum(map(len, dataset))
        self.num_classes = num_classes
        self.image_params = ImageParams(sz=self.sz, nc=self.nc, n=self.n)
        self.nag = NAG(nag_params, self.image_params, rn, resume, num_classes=num_classes, resolution=resolution,
                       datasets=dataset)
        print(f"num classes = {num_classes}")
        print(f"dataset size = {self.n}")

    def create_dirs(self, resume):
        if not os.path.isdir("runs"):
            os.mkdir("runs")
        if not resume:
            shutil.rmtree("runs/ims_%s", ignore_errors=True)
            os.mkdir("runs/ims_%s" % self.rn)
            shutil.rmtree("nets", ignore_errors=True)
        if not os.path.isdir("runs/ims_%s" % self.rn):
            os.mkdir("runs/ims_%s" % self.rn)
        if not os.path.isdir("runs/nets_%s" % self.rn):
            os.mkdir("runs/nets_%s" % self.rn)

    def train_nag(self, opt_params):
        self.nag.train(self.data_tup, opt_params)

    def train_test_nag(self, opt_params):
        self.nag.train_long(self.data_tup[0], self.data_tup[1], self.data_tup[2], resolution=self.sz[0],
                            opt_params=opt_params)
