import os
from pickle import FALSE
import sys
from collections.abc import Iterable
import importlib
import hashlib

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F
from baselines import *
from utils.set_distance import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))


class PointCloudAttack(object):
    def __init__(self, args):
        """WAAttack\SubAttack.
        """
        self.args = args
        self.device = args.device

        self.eps = args.eps
        self.normal = args.normal
        self.step_size = args.step_size
        self.num_class = args.num_class
        self.max_steps = args.max_steps
        self.top5_attack = args.top5_attack

        self.attack_method = args.transfer_attack_method

        self.build_models()
        self.defense_method = args.defense_method
        if not args.defense_method is None and args.defense_method != 'pointcvar':
            self.pre_head = self.get_defense_head(args.defense_method)


    def build_models(self):
        """Build white-box surrogate model and black-box target model.
        """
        # load white-box surrogate models
        MODEL = importlib.import_module(self.args.surrogate_model)
        wb_classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        wb_classifier = wb_classifier.to(self.device)
        # load black-box target models
        MODEL = importlib.import_module(self.args.target_model)
        classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        classifier = classifier.to(self.args.device)
        # load model weights
        wb_classifier = self.load_models(wb_classifier, self.args.surrogate_model)
        classifier = self.load_models(classifier, self.args.target_model)
        # set eval
        self.wb_classifier = wb_classifier.eval()
        self.classifier = classifier.eval()


    def load_models(self, classifier, model_name):
        """Load white-box surrogate model and black-box target model.
        """
        model_path = os.path.join('./checkpoint/' + self.args.dataset, model_name)
        if os.path.exists(model_path + '.pth'):
            checkpoint = torch.load(model_path + '.pth')
        elif os.path.exists(model_path + '.t7'):
            checkpoint = torch.load(model_path + '.t7')
        elif os.path.exists(model_path + '.tar'):
            checkpoint = torch.load(model_path + '.tar')
        else:
            raise NotImplementedError

        try:
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state'])
            else:
                classifier.load_state_dict(checkpoint)
        except:
            classifier = nn.DataParallel(classifier)
            classifier.load_state_dict(checkpoint)
        return classifier


    def CWLoss(self, logits, target, kappa=0, tar=False, num_classes=40):
        """Carlini & Wagner attack loss. 

        Args:
            logits (torch.cuda.FloatTensor): the predicted logits, [1, num_classes].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())

        real = torch.sum(target_one_hot*logits, 1)
        if not self.top5_attack:
            ### top-1 attack
            other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
        else:
            ### top-5 attack
            other = torch.topk((1-target_one_hot)*logits - (target_one_hot*10000), 5)[0][:, 4]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))


    def run(self, points, target):
        """Main attack method.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        if self.attack_method == 'WAAttack':
            return self.WAAttack(points, target)
        elif self.attack_method == 'SubAttack':
            return self.SubAttack(points, target)
        else:
            NotImplementedError


    def get_defense_head(self, method):
        """Set the pre-processing based defense module.

        Args:
            method (str): defense method name.
        """
        if method == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
        elif method == 'srs':
            pre_head = SRSDefense(drop_num=500)
        elif method == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
        else:
            raise NotImplementedError
        return pre_head


    def WAAttack(self, points, target):
        """WAAttack"""
        points = points[:,:,:3].data
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)

        chamfer_loss = ChamferDistance()
        hausdorff_loss = HausdorffDistance()
        clip_func = ClipPointsLinf(budget=self.eps)

        step_size = self.step_size              
        loss_init = 0.0 

        with torch.no_grad():
            original_logits = self.wb_classifier(points.transpose(1, 2))
            original_pred = torch.argmax(original_logits, dim=1).item()
            if original_pred != target:
                return points, original_pred

        current_points = points.clone()

        for it in range(self.max_steps):
            current_points = current_points.detach().requires_grad_(True)
            logits = self.wb_classifier(current_points.transpose(1, 2)) if self.defense_method is None \
                else self.wb_classifier(self.pre_head(current_points.transpose(1, 2)))
            loss1 = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            loss_old = loss1.item()
            if it == 0:
                loss_init = loss_old
            self.wb_classifier.zero_grad()
            loss1.backward()
            grad = current_points.grad.detach()

            norm_max = torch.max(torch.abs(grad.view(grad.shape[0], -1)), dim=1, keepdim=True)[0].view(-1, 1, 1)
            current_points = current_points - step_size * grad / (norm_max + 1e-9) 

            current_points = clip_func(current_points, ori_points)

            next_logits = self.wb_classifier(current_points.transpose(1, 2)) if self.defense_method is None \
                else self.wb_classifier(self.pre_head(current_points.transpose(1, 2)))
            loss2 = self.CWLoss(next_logits, target, kappa=0., tar=False, num_classes=self.num_class)
            loss_new = loss2.item()
            pred = loss_old - loss_new
            rho = pred / (loss_init + 1e-12)

            if rho > 2/(self.max_steps):
                step_size = step_size
            elif rho <= 0:
                step_size = step_size * 0.8
            else:
                step_size = step_size * 1.6

            with torch.no_grad():
                if not self.defense_method is None:
                    check_logits = self.classifier(self.pre_head(current_points.transpose(1, 2)))
                else:
                    check_logits = self.classifier(current_points.transpose(1, 2))
                check_pred = torch.argmax(check_logits, dim=1).item()

                if check_pred != target:
                    break

        return current_points, check_pred


    def SubAttack(self, points, target, n=4):
        current_points = points[:,:,:3]
        ori_points = points[:,:,:3]
        clip_func = ClipPointsLinf(budget=self.eps)

        step_size = self.step_size               
        loss_init = 0.0 
        
        for it in range(self.max_steps):
            subset_indices = self.split_point_cloud_random(current_points, n)[0]

            best_combination, best_score = self.find_best_synergistic_subset_combination(
                current_points, target, subset_indices, n, step_size, clip_func
            )
            
            cumulative_mask = torch.zeros_like(subset_indices, dtype=torch.bool)
            for subset_idx in best_combination:
                mask = (subset_indices == subset_idx)
                cumulative_mask = cumulative_mask | mask
            
            current_points = current_points.detach().requires_grad_(True)
            
            logits = self.wb_classifier(current_points.transpose(1, 2)) if self.defense_method is None \
                else self.wb_classifier(self.pre_head(current_points.transpose(1, 2)))
            loss_old = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            
            if it == 0:
                loss_init = loss_old
            self.wb_classifier.zero_grad()
            loss_old.backward()
            grad = current_points.grad.data
            subset_grad = grad[0][cumulative_mask] 
            norm = torch.max(torch.abs(subset_grad.view(-1)), dim=0)[0]
            
            with torch.no_grad():
                current_points[0][cumulative_mask] = current_points[0][cumulative_mask] - step_size * subset_grad / (norm + 1e-9)
                current_points = clip_func(current_points, ori_points)
                

            next_logits = self.wb_classifier(current_points.transpose(1, 2)) if self.defense_method is None \
                else self.wb_classifier(self.pre_head(current_points.transpose(1, 2)))
            loss2 = self.CWLoss(next_logits, target, kappa=0., tar=False, num_classes=self.num_class)
            loss_new = loss2.item()
            pred = loss_old - loss_new
            rho = pred / (loss_init + 1e-12)

            if rho > 2/(self.max_steps):
                step_size = step_size
            elif rho <= 0:
                step_size = step_size * 0.8
            else:
                step_size = step_size * 1.6
        
            with torch.no_grad():
                if not self.defense_method is None:
                    check_logits = self.classifier(self.pre_head(current_points.transpose(1, 2)))
                else:
                    check_logits = self.classifier(current_points.transpose(1, 2))
                pred = torch.argmax(check_logits, dim=1).item()
                if pred != target:
                    break
        
        return current_points, pred


    def split_point_cloud_random(self, points, num_subsets):
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        batch_subset_indices = []
        for b in range(batch_size):
            random_indices = torch.randperm(num_points, device=points.device)
            points_per_subset = num_points // num_subsets
            remainder = num_points % num_subsets
            subset_indices = torch.zeros(num_points, dtype=torch.long, device=points.device)
            start_idx = 0
            for subset_idx in range(num_subsets):
                current_subset_size = points_per_subset + (1 if subset_idx < remainder else 0)
                end_idx = start_idx + current_subset_size
                subset_random_indices = random_indices[start_idx:end_idx]
                subset_indices[subset_random_indices] = subset_idx
                start_idx = end_idx
            batch_subset_indices.append(subset_indices)
        
        return batch_subset_indices


    def find_best_synergistic_subset_combination(self, points, target, subset_indices, n, step_size, clip_func):
        best_combination = list(range(n))
        best_score = 0.0
        
        chamfer_loss = ChamferDistance()
        hausdorff_loss = HausdorffDistance()
        
        with torch.no_grad():
            current_logits = self.wb_classifier(points.transpose(1, 2)) if self.defense_method is None \
                else self.wb_classifier(self.pre_head(points.transpose(1, 2)))
            current_loss = self.CWLoss(current_logits, target, kappa=0., tar=False, num_classes=self.num_class).item()
        points_grad = points.clone().detach().requires_grad_(True)
        logits = self.wb_classifier(points_grad.transpose(1, 2)) if self.defense_method is None \
            else self.wb_classifier(self.pre_head(points_grad.transpose(1, 2)))
        loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
        
        self.wb_classifier.zero_grad()
        loss.backward()
        grad = points_grad.grad.data
        from itertools import combinations

        
        for r in range(1, n+1):
            for subset_combination in combinations(range(n), r):
                cumulative_mask = torch.zeros_like(subset_indices, dtype=torch.bool)
                for subset_idx in subset_combination:
                    mask = (subset_indices == subset_idx)
                    cumulative_mask = cumulative_mask | mask
                
                if cumulative_mask.sum() == 0:
                    continue
                
                subset_grad = grad[0][cumulative_mask]
                norm = torch.max(torch.abs(subset_grad.view(-1)), dim=0)[0]
                new_points = points.clone()
                with torch.no_grad():
                    new_points[0][cumulative_mask] = new_points[0][cumulative_mask] - step_size * subset_grad / (norm + 1e-9)
                    new_points = clip_func(new_points, points)
                    
                    new_logits = self.wb_classifier(new_points.transpose(1, 2)) if self.defense_method is None \
                        else self.wb_classifier(self.pre_head(new_points.transpose(1, 2)))
                    new_loss = self.CWLoss(new_logits, target, kappa=0., tar=False, num_classes=self.num_class).item()
                    
                loss_reduction = current_loss - new_loss
                    
                subset_chamfer = chamfer_loss(new_points, points).item()
                subset_hausdorff = hausdorff_loss(new_points, points).item()
                subset_l2 = torch.sqrt(((new_points - points)**2).sum(1).sum(1)).item()
                distortion_metrics = 1000 * subset_chamfer + 100 * subset_hausdorff + 0.1 * subset_l2
                combined_score = loss_reduction - 0.1 * distortion_metrics
                if combined_score >= best_score:
                    best_score = combined_score
                    best_combination = list(subset_combination)
    
        return best_combination, best_score

