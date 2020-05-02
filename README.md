# domain_adaptation_project
This is the project github for 'popular approach to domain adaptation'.

We reimplement 'Maximum Classifier Discrepancy for Unsupervised Domain Adaptation'(MCD),'ADVERSARIAL DROPOUT REGULARIZATION'(ADR)
' and 'Semi-supervised Domain Adaptation via Minimax Entropy' in pytorch. 

We mainly carried out experiments in digit dataset(USPS,SVHN,MNIST) and VisDA 2017.


To perform experiments in digit datasets, 
Run 'classfication/solver.py'
Run 'classfication/solver.py --source='source' --target='target' to change the target domain and source domain'
Run 'classfication/solver.py --mode='ad-drop' to use ADR method
Run 'classfication/solver.py --mode='normal' to use MCD method


To perform experiment in VisDA dataset,
Run 'VisDA/solver.py'
Run 'VisDA/solver.py --mode='ad-drop' to use ADR method
Run 'VisDA/solver.py --mode='normal' to use MCD method
