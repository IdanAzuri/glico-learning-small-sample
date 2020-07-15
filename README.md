

# Generative Latent Implicit Conditional Optimization when Learning from Small Sample
If you find this repository useful in your research, please cite the following paper:



## 1. Requirements
* torch>= 1.3.0

* torchvision>=0.4.2

* easyargs


```console
dir=path-to-repo/learning-from-small-sample/glico_model
cd $dir
```


## 2. Datasets

The following datasets have been used in the paper:

- [Caltech UCSD Birds-200-2011][1] (CUB)
- [CIFAR-100][2] (CIFAR-100)

To experiment with differently sized variants of the CUB dataset, download the [modified image list files][3] files and unzip the obtained archive into the root directory of your CUB dataset
## 3.  Multiple shots on CUB



```console
UNLABELED=10
SEED=0

for SHOTS in 5 10 20 30; do
echo "glico CUB  samples per class: $SHOTS"
  # train

  s=" train_glico.py --rn d128_v2_0.001bn --d conv --pixel --z_init rndm --resume --tr --data cub --dim 128 --epoch 202 --fewshot --shot ${SHOTS} --seed ${SEED}"
  python3 $s
  echo $s

  sleep 15

  # eval

  s="evaluation.py -d resnet50 --pretrained --keyword  cub_d128_v2_0.001bn_pixel_classifier_tr_fs_${SHOTS}  --is_inter  --augment --epoch 200 --data cub  --fewshot --shot ${SHOTS} --dim 128 --seed ${SEED}"
  echo $s
  python3 $s
done
```
## 4. Multiple shots on CIFAR-100
```console

UNLABELED=10
SEED=0


for SHOTS in 10 25 50 100; do
  echo "glico CIFAR100 samples per classt: $SHOTS"
  # train

  s="train_glico.py --rn  sgd_${UNLABELED}unsuprvised --fewshot --shot $SHOTS --d conv --pixel  --z_init rndm --resume --unlabeled_shot ${UNLABELED} --epoch 202 --noise_proj --tr --seed ${SEED}"
  echo $s
  python3 $s

  sleep 15

  # eval

  s="evaluation.py -d wideresnet --keyword cifar-100_sgd_${UNLABELED}unsuprvised_pixel_classifier_tr_fs_${SHOTS}_ce_noise_proj --is_inter --augment --epoch 200 --data cifar --pretrained --fewshot --shot $SHOTS --unlabeled_shot ${UNLABELED} --loss_method ce --seed ${SEED}"
  echo $s
  python3 $s
done
```

## 5. Baseline for CIFAR-100 
* Try the different flags: <br /> 
--random_erase<br /> 
--cutout<br /> 
--autoaugment <br /> 
or none of the above fro 'clean' baseline
* Choose the classifier architecture from the following: 
<br /> --d widerenset
<br /> --d resnet50
<br /> --d resnet (resnet110)
<br /> --d vgg (vgg19)
 ```
SHOTS=50
UNLABEL=1
SEED=0
echo " Baseline CIFAR random_erase shot: $SHOTS"
 s=" baseline_classification.py --epoch 200 -d wideresnet --augment --data cifar  --fewshot --shot  $SHOTS --unlabeled_shot 10 --seed ${SEED}"
echo $s
python3 $s
echo " Baseline CIFAR random_erase shot: $SHOTS"

```
[1]: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
[2]: https://www.cs.toronto.edu/~kriz/cifar.html
[3]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.2.0/cub-subsampled-splits.zip
