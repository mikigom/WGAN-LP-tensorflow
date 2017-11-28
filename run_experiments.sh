#!/usr/bin/env bash

# Fig6-Top
python3 trainer.py --Regularization_type GP --Purturbation_type wgan_gp --Lambda 10.0
# Fig6-Middle, Fig13, Fig15
python3 trainer.py --Regularization_type GP --Purturbation_type wgan_gp --Lambda 1.0
# Fig6-Bottom
python3 trainer.py --Regularization_type LP --Purturbation_type wgan_gp --Lambda 10.0
# Not shown in Fig6, but written
python3 trainer.py --Regularization_type LP --Purturbation_type wgan_gp --Lambda 1.0
python3 trainer.py --Regularization_type LP --Purturbation_type wgan_gp --Lambda 100.0

# Fig11-Top
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type GP --Purturbation_type wgan_gp --Lambda 10.0
# Fig11-Middle
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type GP --Purturbation_type wgan_gp --Lambda 1.0
# Fig11-Bottom
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 10.0
# Not shown in Fig11, but written
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 1.0
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 100.0

# Fig12-Top
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type GP --Purturbation_type wgan_gp --Lambda 10.0
# Fig12-Middle
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type GP --Purturbation_type wgan_gp --Lambda 1.0
# Fig12-Bottom
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 10.0
# Not shown in Fig12, but written
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 1.0
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 100.0

# Fig7-Top, Fig9-Top
python3 trainer.py --Regularization_type GP --Purturbation_type wgan_gp --Lambda 5.0
# Fig7-Bottom, Fig9-Bottom
python3 trainer.py --Regularization_type LP --Purturbation_type wgan_gp --Lambda 5.0

# Fig8-Top
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 5.0
# Fig8-Middle, Fig14-Top
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_both --Lambda 5.0
# Fig8-Bottom, Fig14-Bottom
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_both --Lambda 5.0


python3 trainer.py --n_epoch 2000 --Regularization_type GP --Purturbation_type wgan_gp --Lambda 1.0 --emd_records True
python3 trainer.py --n_epoch 2000 --Regularization_type GP --Purturbation_type wgan_gp --Lambda 5.0 --emd_records True
python3 trainer.py --n_epoch 2000 --Regularization_type LP --Purturbation_type wgan_gp --Lambda 5.0 --emd_records True
python3 trainer.py --n_epoch 2000 --Regularization_type GP --Purturbation_type dragan_both --Lambda 5.0 --emd_records True
python3 trainer.py --n_epoch 2000 --Regularization_type LP --Purturbation_type dragan_both --Lambda 5.0 --emd_records True
