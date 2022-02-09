#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from pathlib import Path
import uuid
import tempfile
import math
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Tuple, Dict, Any
import torch
from fire import Fire
from ignite.contrib.metrics import ROC_AUC
from ignite.contrib.handlers import ProgressBar, create_lr_scheduler_with_warmup, CosineAnnealingScheduler
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    global_step_from_engine,
)
from ignite.metrics import Loss, Precision, Recall, RunningAverage, Accuracy
from loguru import logger
import sys
import numpy as np
import torch
import yaml
import utils
import models
import dataset
import losses

logger.configure(handlers=[{
    "sink": sys.stderr,
    "format": "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}",
    'level': 'DEBUG',
}])

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device(DEVICE)


def transfer_to_device(batch):
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def _mixup_weights(size, alpha):
    return torch.tensor(np.random.beta(alpha, alpha, size=size),
                        device=DEVICE,
                        dtype=torch.float32)


class Runner(object):

    def __init__(self, seed=42):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info(f"Using seed {seed}")

    def __setup_train(self,
                      config: Path,
                      default_args=utils.DEFAULT_TRAIN_ARGS,
                      **override_kwargs) -> Dict[str, Any]:
        config_parameters = utils.parse_config_or_kwargs(
            config, default_args=default_args, **override_kwargs)
        outputdir = Path(
            Path(config_parameters['outputpath']) / Path(config).stem /
            f"{config_parameters['student']}_{config_parameters['teacher']}",
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m')}_{uuid.uuid1().hex}"
        )
        outputdir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Storing output in {outputdir}")
        log_fname = config_parameters.get('logfile', 'train.log')
        output_log = outputdir / log_fname
        logger.add(
            output_log,
            enqueue=True,
            level='INFO',
            format=
            "[<red>{level}</red> <green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
        )

        student = getattr(models, config_parameters['student'])(
            outputdim=527,
            spectransforms=utils.parse_spectransforms(
                config_parameters['spectransforms']),
            wavtransforms=utils.parse_wavtransforms(
                config_parameters['wavtransforms']),
            **config_parameters['student_args'])
        teacher = getattr(models, config_parameters['teacher'])(
            outputdim=527, **config_parameters['teacher_args'])
        # pretrained_teacher = torch.load(
        # config_parameters['pretrained_teacher'], map_location='cpu')
        pretrained_teacher_path = config_parameters['pretrained_teacher']
        if 'http' in pretrained_teacher_path:
            pretrained_teacher = torch.hub.load_state_dict_from_url(
                config_parameters['pretrained_teacher']
            )
        else:
            pretrained_teacher = torch.load(pretrained_teacher_path,map_location='cpu')
        if 'model' in pretrained_teacher:
            pretrained_teacher = pretrained_teacher['model']
        teacher = models.load_pretrained(teacher, pretrained_teacher)
        if config_parameters.get('pretrained'):
            logger.info(
                f"Loading pretrained student model {config_parameters['pretrained']}"
            )
            pretrained_model = torch.load(config_parameters['pretrained'],
                                          map_location='cpu')
            if 'model' in pretrained_model:
                pretrained_model = pretrained_model['model']
            student = models.load_pretrained(student, pretrained_model)
        for k, v in config_parameters.items():
            logger.info(f"{k} : {v}")

        return {
            'outputdir': outputdir,
            'params': config_parameters,
            'student': student,
            'teacher': teacher,
        }

    def __train(self,
                model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                cv_dataloader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer,
                outputpath: Path,
                config_parameters,
                train_fun=None):
        model.to(DEVICE)
        mixup_alpha = config_parameters.get('mixup_alpha')
        epochs = config_parameters.get('epochs', 100)
        early_stop = config_parameters.get('early_stop', 5)
        epoch_length = config_parameters.get('epoch_length')
        doscheduler = config_parameters.get('use_scheduler', None)

        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                x, y, _ = transfer_to_device(batch)
                mixup_lambda = None
                if mixup_alpha and mixup_alpha > 0:
                    mixup_lambda = _mixup_weights(len(x) // 2, mixup_alpha)
                    x = utils.mixup(x, mixup_lambda)
                    y = utils.mixup(y, mixup_lambda)
                output_student, _ = model(x)
                suploss = criterion(output_student, y)
                loss = suploss
                loss.backward()
                optimizer.step()
                return {
                    'total_loss': loss.item(),
                    'sup_loss': suploss.item(),
                }

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                data, targets, filenames = transfer_to_device(batch)
                clip_out, _ = model(data)
                return clip_out, targets

        def compute_metrics(engine):
            inference_engine.run(cv_dataloader)
            results = inference_engine.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
            ] + [f"{metric} {results[metric]:<5.2f}" for metric in results
                 ] + [f"{optimizer.param_groups[0]['lr']}"]
            logger.info(" ".join(output_str_list))

        train_engine = Engine(_train_batch if train_fun is None else train_fun)
        inference_engine = Engine(_inference)
        evaluation_metrics = {
            'dprime': losses.DPrime(),
            'AuC': losses.ROC_AUC(),
            'Loss': Loss(criterion),
            'mAP': losses.mAP(),
        }
        for name, metric in evaluation_metrics.items():
            metric.attach(inference_engine, name)
        checkpoint_saver = Checkpoint(
            {
                'model': model,
                'config': utils.DictWrapper(config_parameters)
            },
            DiskSaver(outputpath),
            n_saved=config_parameters.get('n_saved', 1),
            global_step_transform=global_step_from_engine(train_engine),
            filename_prefix='best',
            score_function=Checkpoint.get_default_score_fn('mAP', 1.0),
            score_name='mAP',
        )
        inference_engine.add_event_handler(Events.COMPLETED, checkpoint_saver)
        ProgressBar().attach(train_engine,
                             output_transform=lambda x: {
                                 'Total': x['total_loss'],
                                 'Sup': x['sup_loss'],
                                 'TS': x['ts_loss'],
                             })
        decay_steps = epochs * len(
            train_dataloader
        ) if epoch_length == None else epochs * epoch_length
        if doscheduler:
            logger.info(f"Decaying with {decay_steps} steps")
            scheduler = CosineAnnealingScheduler(optimizer,
                                                 decay_steps=decay_steps,
                                                 final_lrs=0.0)
            scheduler = create_lr_scheduler_with_warmup(scheduler,
                                                        warmup_start_value=0.0,
                                                        warmup_duration=1000)
            train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
        earlystop_handler = EarlyStopping(
            patience=early_stop,
            score_function=lambda engine: engine.state.metrics['mAP'],
            trainer=train_engine)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics)
        inference_engine.add_event_handler(Events.COMPLETED, earlystop_handler)

        train_engine.run(
            train_dataloader,
            max_epochs=epochs,
            epoch_length=epoch_length,
        )
        return checkpoint_saver.last_checkpoint

    def train(self, config, **override_kwargs):
        setup_params = self.__setup_train(
            config, default_args=utils.DEFAULT_CHUNK_ARGS, **override_kwargs)

        config_parameters = setup_params['params']
        outputdir = setup_params['outputdir']
        student = setup_params['student']
        teacher = setup_params['teacher']
        data_bs = config_parameters['batch_size']

        train_df = utils.read_tsv_data_chunked(
            config_parameters['train_data'],
            chunk_length=config_parameters['chunk_length'],
            chunk_hop=config_parameters.get('chunk_hop', None))
        cv_df = utils.read_tsv_data(config_parameters['cv_data'])

        train_labeled_dataset = dataset.WeakChunkedHDF5Dataset(train_df,
                                                               num_classes=527)
        if config_parameters.get('sampler', 'balanced'):
            sampler_kwargs = {
                'sampler': dataset.BalancedSampler(train_df['labels'].values),
                'shuffle': False
            }
        else:
            sampler_kwargs = {'shuffle': True}
        logger.info(f"Using sampler {sampler_kwargs}")
        traindataloader = torch.utils.data.DataLoader(
            train_labeled_dataset,
            batch_size=data_bs,
            num_workers=2,
            collate_fn=dataset.sequential_pad,
            persistent_workers=True,
            **sampler_kwargs)

        cv_dataset = dataset.WeakHDF5Dataset(cv_df, num_classes=527)
        cvdataloader = torch.utils.data.DataLoader(
            cv_dataset,
            batch_size=data_bs,
            collate_fn=dataset.sequential_pad,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
            num_workers=2)
        logger.debug(
            f"Train data size {len(traindataloader)}, CV data size {len(cvdataloader)}"
        )

        criterion = getattr(losses, config_parameters['loss'])(
            **config_parameters['loss_args']).to(DEVICE)
        consistency_criterion = getattr(
            losses, config_parameters['consistency_criterion'])().to(DEVICE)
        logger.debug(
            f"Criterion: {criterion} Consistency Criterion: {consistency_criterion} "
        )

        optimizer = getattr(
            torch.optim,
            config_parameters['optimizer'],
        )(student.parameters(), **config_parameters['optimizer_args'])
        student.to(DEVICE)
        teacher.to(DEVICE)
        mixup_alpha = config_parameters.get('mixup_alpha', None)
        logger.info(f"{mixup_alpha=}")

        #init optimizer
        # Estimate first model
        logger.info(
            f"Beginning Training with model {student.__class__.__name__} in {outputdir}"
        )
        ts_alpha = config_parameters.get('ts_alpha', None)

        get_alpha = lambda x: 0.5  # Default, just average of the two losses
        if isinstance(ts_alpha, int) or isinstance(ts_alpha, float):
            get_alpha = lambda x: ts_alpha

        train_fun = config_parameters.get('train_fun', None)
        logger.info(
            f"Using training function {train_fun} and teacher student alpha: {ts_alpha}"
        )

        def train_batch(engine, batch):
            student.train()
            teacher.eval()
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                x, y, _ = transfer_to_device(batch)
                mixup_lambda = None
                if mixup_alpha and mixup_alpha > 0:
                    mixup_lambda = _mixup_weights(len(x) // 2, mixup_alpha)
                    x = utils.mixup(x, mixup_lambda)
                    y = utils.mixup(y, mixup_lambda)
                with torch.no_grad():
                    output_teacher, _ = teacher(x)
                    #Change to soft (default) or hard (with label_type == hard)
                    output_teacher = output_teacher.detach()
                output_student, _ = student(x)
                suploss = criterion(output_student, y)
                ts_loss = consistency_criterion(output_student, output_teacher)
                alpha = get_alpha(engine.state.epoch)
                loss = alpha * ts_loss + (1. - alpha) * suploss
                loss.backward()
                optimizer.step()
                return {
                    'total_loss': loss.item(),
                    'sup_loss': suploss.item(),
                    'ts_loss': ts_loss.item(),
                }

        saved_params_file = self.__train(model=student,
                                         train_dataloader=traindataloader,
                                         cv_dataloader=cvdataloader,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         train_fun=train_batch,
                                         outputpath=outputdir,
                                         config_parameters=config_parameters)
        return outputdir / saved_params_file

    def run(self, config, **override_kwargs):
        result_model = self.train(config, **override_kwargs)
        self.evaluate(result_model)

    def evaluate(
        self,
        experiment_path: Path,
        test_data_file: Path = 'data/labels/eval.csv',
        label_indices:
        Path = 'data/csvs/class_labels_indices.csv',
        num_classes =527,
        **kwargs,
    ):
        experiment_path = Path(experiment_path)
        test_data_file = Path(test_data_file)
        if experiment_path.is_file():  # Best model passed!
            training_dump = torch.load(experiment_path, map_location='cpu')
            experiment_path = experiment_path.parent  # Just set upper path as default
        else:
            training_dump = torch.load(next(
                experiment_path.glob("*checkpoint*")),
                                       map_location='cpu')
        config_parameters = training_dump['config']
        student = config_parameters['student']
        trained_model_params = training_dump['model']

        model = getattr(models, student)(outputdim=num_classes,
                                         **config_parameters['student_args'])
        model = models.load_pretrained(model,
                                       trained_model_params).to(DEVICE).eval()

        eval_df = utils.read_tsv_data(test_data_file)
        eval_dataset = dataset.WeakHDF5Dataset(eval_df, num_classes=527)
        evaldataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config_parameters['batch_size'],
            shuffle=False,
            collate_fn=dataset.sequential_pad,
            num_workers=2)

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                data, targets, filenames = transfer_to_device(batch)
                clip_out, _ = model(data)
                return clip_out, targets

        def _output_transform(output):
            y_pred, y = output
            return (y_pred > 0.2).float(), y

        eval_engine = Engine(_inference)
        evaluation_metrics = {
            'dprime': losses.DPrime(),
            'AuC': losses.ROC_AUC(),
            'ap': losses.mAP(average=None),
            'Recall@0.2': Recall(_output_transform)
        }
        for name, metric in evaluation_metrics.items():
            metric.attach(eval_engine, name)
        ProgressBar().attach(eval_engine)
        logger.info(f"Running evaluation on {test_data_file}")
        eval_engine.run(evaldataloader)

        ap = eval_engine.state.metrics['ap']
        auc = eval_engine.state.metrics['AuC']
        dprime = eval_engine.state.metrics['dprime']
        recall_02 = eval_engine.state.metrics['Recall@0.2']

        logger.add(experiment_path /
                   f'tagging_preds_{test_data_file.stem}.txt',
                   format='{message}',
                   level='INFO',
                   mode='w')

        idx_to_name = {idx: idx for idx in range(num_classes)}
        pad_length = 4  # Default padding for 3-4 digit numbers
        if label_indices != None and Path(label_indices).exists():
            classmaps = pd.read_csv(label_indices)
        # Get proper class names
        idx_to_name = classmaps.set_index('index')['display_name'].to_dict()
        pad_length = max(classmaps['display_name'].apply(len))
        sorted_idxs = np.argsort(ap)[::-1]
        logger.info(f"Classes performance (AP)")
        for i, idx in enumerate(sorted_idxs):
            logger.info(
                f"Top-{i+1} Label {idx_to_name[idx]:<{pad_length}}:{ap[idx]*100:.2f}",
            )
        logger.info(f"mAP:{ap.mean()*100:.3f}")
        logger.info(f"AUC:{auc.mean()*100:.3f}")
        logger.info(f"Dprime:{dprime:.3f}")
        logger.info(f"Rec@0.2:{recall_02*100:.3f}")


if __name__ == "__main__":
    Fire(Runner)
