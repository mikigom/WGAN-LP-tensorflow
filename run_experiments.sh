#!/usr/bin/env bash
# 8-1
# Fig6-1
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 10.0
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_both --Lambda 10.0
# Fig6-2
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 1.0
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_both --Lambda 1.0
# Fig6-3
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 10.0
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_both --Lambda 10.0
# Not shown in Fig6, but written
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 1.0
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_both --Lambda 1.0

# B.1
# Fig6-1
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 10.0
python3 trainer.py --dataset GeneratorGaussians8  --Regularization_type GP --Purturbation_type dragan_both --Lambda 10.0
# Fig6-2
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 1.0
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type GP --Purturbation_type dragan_both --Lambda 1.0
# Fig6-3
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 10.0
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type dragan_both --Lambda 10.0
# Not shown in Fig6, but written
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 1.0
python3 trainer.py --dataset GeneratorGaussians8 --Regularization_type LP --Purturbation_type dragan_both --Lambda 1.0

# Fig6-1
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 10.0
python3 trainer.py --dataset GeneratorGaussians25  --Regularization_type GP --Purturbation_type dragan_both --Lambda 10.0
# Fig6-2
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 1.0
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type GP --Purturbation_type dragan_both --Lambda 1.0
# Fig6-3
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 10.0
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type dragan_both --Lambda 10.0
# Not shown in Fig6, but written
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 1.0
python3 trainer.py --dataset GeneratorGaussians25 --Regularization_type LP --Purturbation_type dragan_both --Lambda 1.0

# 8-2 (and Fig8)
# Fig7-1
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_only_training --Lambda 5.0
python3 trainer.py --Regularization_type GP --Purturbation_type dragan_both --Lambda 5.0
# Fig7-2
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 5.0
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_both --Lambda 5.0
# Not shown in Fig6, but written
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_only_training --Lambda 100.0
python3 trainer.py --Regularization_type LP --Purturbation_type dragan_both --Lambda 100.0