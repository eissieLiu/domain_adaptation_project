# domain_adaptation_project
This is the project GitHub for **popular approach to domain adaptation**.

We reimplement 
* 'Maximum Classifier Discrepancy for Unsupervised Domain Adaptation' **(MCD)** [2] ,
* 'Adversarial Dropout Regularization'**(ADR)** [1] 
* 'Semi-supervised Domain Adaptation via Minimax Entropy' [3] 

in *pytorch*. 

We mainly carried out experiments in **digit dataset** (USPS,SVHN,MNIST) and **VisDA 2017**.

## Experiments in digit datasets
* Run `python classfication/solver.py`
* Run `python classfication/solver.py --source="source" --target="target"` to change the target domain and source domain
* Run `python classfication/solver.py --mode='ad-drop` to use ADR method
* Run `python classfication/solver.py --mode='normal' `to use MCD method
* Run `python MME/main.py` to use Minimax Entropy
* Run `python MME/main.py --source="source" --target="target"` to use Minimax Entropy and change the target domain and source domain

## Experiments in VisDA datasets
* Run `python VisDA/solver.py`
* Run `python VisDA/solver.py --mode='ad-drop'` to use ADR method
* Run `python VisDA/solver.py --mode='normal'` to use MCD method


## Reference
1. Kuniaki Saito, Yoshitaka Ushiku, Tatsuya Harada, and Kate Saenko. Adversarial dropout regularization. In ICLR, 2018.<br>
2. Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, and Tatsuya Harada. Maximum classifier discrepancy for unsupervised domain adaptation. In CVPR, 2018.<br>
3. Saito, Kuniaki, et al. "Semi-supervised domain adaptation via minimax entropy." Proceedings of the IEEE International Conference on Computer Vision. 2019.<br>

