import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
from models.utils import build_common_model
from lib.utils import *
from lib.post_process import*

import torchcp
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import SAPS,APS
from torchcp.classification.utils.metrics import Metrics

metrics = Metrics()
class experiment:
    def __init__(self,model_name,alpha,predictor,dataset_name,post_hoc,num_trials) -> None:
        self.model_name = model_name
        self.alpha=  alpha
        self.predictor = predictor
        self.dataset_name = dataset_name
        if self.dataset_name =="imagenet":
            self.num_calsses = 1000
        else:
            raise NotImplementedError
        self.post_hoc =  post_hoc
        self.model = build_common_model(self.model_name,dataset_name)

        ### Data Loading
        self.dataset = get_logits_dataset(self.model_name,self.dataset_name)
        
        # trials
        self.num_trials = num_trials
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def run(self, n_data_conf, n_data_val, pct_paramtune, bsz):
        ### Perform experiment
        top1s = np.zeros((self.num_trials ,))
        top5s = np.zeros((self.num_trials ,))
        coverages = np.zeros((self.num_trials ,))
        sizes = np.zeros((self.num_trials ,))

        for i in tqdm(range(self.num_trials )):
            self.seed  = i
            self._fix_randomness(self.seed)
            top1_avg, top5_avg, cvg_avg, sz_avg  = self.trial( n_data_conf, n_data_val, pct_paramtune, bsz)
            top1s[i] = top1_avg
            top5s[i] = top5_avg
            coverages[i] = cvg_avg
            sizes[i] = sz_avg
            print(f'\n\tTop1: {np.median(top1s[0:i+1]):.3f}, Top5: {np.median(top5s[0:i+1]):.3f}, Coverage: {np.median(coverages[0:i+1]):.3f}, Size: {np.median(sizes[0:i+1]):.3f}\033[F', end='')
        print('')
    



    def trial(self, n_data_conf, n_data_val, pct_paramtune, bsz):
        alpha = self.alpha
        cal_dataset, val_dataset = split2(self.dataset, n_data_conf, n_data_val)
        cal_logits = [x[0]for x in cal_dataset]
        cal_labels = [x[1]for x in cal_dataset]

        val_logits = [x[0] for x in val_dataset]
        val_labels = [x[1] for x in val_dataset]


        score_function = APS()
        prediction_sets = []
        for i in range(val_logits.shape[0]):
            this_logit = val_logits[i]
            prediction_set = []
            for test_label in range(val_logits.shape[1]):
                transformation = OptimalTeamperatureScaling(1.3)
                this_cal_logits = torch.concatenate(cal_logits, torch.Tensor(this_logit))
                this_cal_labels = torch.concatenate(val_labels, torch.Tensor(test_label))
                this_cal_dataset = torch.utils.data.TensorDataset(this_cal_logits, this_cal_labels.long())
                this_cal_loader = torch.utils.data.DataLoader(this_cal_dataset, batch_size=bsz, shuffle=False, pin_memory=True)
                transformation = self.get_optimal_parameters(transformation, this_cal_loader)
                this_cal_logits, this_cal_labels = postHocLogits(transformation, this_cal_loader, self.device, self.num_calsses)
                nonconformity_scores = score_function(this_cal_logits, this_cal_labels)
                test_example_score = nonconformity_scores[-1]
                cal_scores = nonconformity_scores[:-1]
                p_value = torch.sum(test_example_score <= nonconformity_scores) / nonconformity_scores.shape[0]
                # accept the null-hypothesis
                if p_value > alpha:
                    prediction_set.append(test_label)

            prediction_sets.append(prediction_set)



        coverage_rate = metrics('coverage_rate')(prediction_sets, val_labels)
        average_size = metrics('average_size')(prediction_sets, val_labels)
        prec1, prec5 = accuracy(val_logits, val_labels, topk=(1, 5))

        return prec1,prec5,coverage_rate,average_size
    

    def  get_optimal_parameters(self,transformation, calib_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        
        device = self.device
        transformation.to(device)
        max_iters=10
        lr=0.01
        epsilon=0.01
        nll_criterion = nn.CrossEntropyLoss().cuda()

        T = transformation.temperature

        optimizer = optim.SGD([transformation.temperature], lr=lr)
        for iter in range(max_iters):
            T_old = T.item()
            # print(T_old)
            for x, targets in calib_loader:
                optimizer.zero_grad()
                x = x.cuda()
                x.requires_grad = True
                out = x/transformation.temperature
                loss = nll_criterion(out, targets.long().cuda())
                
                loss.backward()
                optimizer.step()
            T = transformation.temperature
            if abs(T_old - T.item()) < epsilon:
                break

        return transformation

    def _fix_randomness(self,seed=0):
        ### Fix randomness 
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates conformal predictors',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset')
    parser.add_argument('--model', type=str, default='ResNeXt101', help='model')
    parser.add_argument('--predictor', type=str, default='SAPS', help='the predictor of CP.')
    parser.add_argument('--alpha', type=float, default=0.1, help='the error rate.')
    parser.add_argument('--trials', type=int, default=5, help='number of trials')
    parser.add_argument('--post_hoc', type=str, default="TS", help='the confidence calibration method.')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    model = args.model
    num_trials = args.trials        
    alpha = args.alpha
    post_hoc = args.post_hoc
    predictor = args.predictor
    
    if dataset_name==  "imagenet":
        n_data_conf = 30000
        n_data_val = 10
    else:
        raise NotImplementedError
    
    pct_paramtune = 0.2
    bsz = 128
    cudnn.benchmark = True
    print(f'Model: {model} | Desired coverage: {1-alpha} | Predictor: {predictor}| Calibration: {post_hoc}')
    this_experiment =  experiment(model,alpha,predictor,dataset_name,post_hoc,num_trials)
    this_experiment.run( n_data_conf, n_data_val, pct_paramtune, bsz) 
