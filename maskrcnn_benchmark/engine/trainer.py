# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from .DAG import DAGrad
import copy
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def update_discriminator_weights(layer_list, FedAvg):
    weight_tmp = []
    for layer in layer_list:
        weight_tmp.append(copy.deepcopy(layer.state_dict()))
    fed_weight = FedAvg(weight_tmp)
    for index,layer in enumerate(layer_list):
        layer.load_state_dict(fed_weight)
    return fed_weight,weight_tmp


def load_discriminator_with_weights(layer_list, fed_weight, weight, raw_weight):
    fed_weight_tmp = {}
    weight=weight.tolist()

    for key in fed_weight.keys():
        fed_weight_tmp[key] = 0

    for key in fed_weight.keys():
        for index, layer in enumerate(layer_list):
            scalar = weight[index]  
            fed_weight_tmp[key] += raw_weight[index][key] * scalar
    for index, layer in enumerate(layer_list):
        layer.load_state_dict(fed_weight_tmp) 

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    val_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg     
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    max_AP50_bank = cfg.SOLVER.INITIAL_AP50
    min_val_step=cfg.SOLVER.MIN_VAL_SETP
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration>=min_val_step:
            if val_data_loader is not None:
                model.eval()
                val_results = validataion(cfg, model, val_data_loader[0], distributed=False)

                meter_AP50 = val_results['map']
                # logger.info('[validation mAP] AP: {}, AP50: {}'.format(meter_AP,meter_AP50))
                logger.info('[validation mAP] AP: {}, AP50: {}'.format(meter_AP50, meter_AP50))

                if meter_AP50 > max_AP50_bank:
                    max_AP50_bank = meter_AP50
                    logger.info('best mAP: {}'.format(max_AP50_bank))
                    checkpointer.save("best_model_updated_{}_{:07d}".format(meter_AP50, iteration), **arguments)
                model.train()
            else:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_da_train(
    model,
    source_data_loader,
    target_data_loader,
    val_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    max_AP50_bank = cfg.SOLVER.INITIAL_AP50
    min_val_step=cfg.SOLVER.MIN_VAL_SETP

    optimizer_new = DAGrad(optimizer) 
    fed_iter=0
    print("\n ubuntu@yunan-3-3:/mnt/d/user1/UniDAOD_ALL-github-v2$ \n")
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
                
        data_time = time.time() - end
        arguments["iteration"] = iteration
        #########################################################
        #########################################################
        """UDA SETTING: remove the target gt label"""
        target_targets[0].bbox= torch.tensor([[0,0,0,0]]).cuda()
        #########################################################
        #########################################################
        scheduler.step()
        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]
        loss_dict = model(images, targets)
        losses_det=loss_dict['loss_objectness']+loss_dict['loss_box_reg']+loss_dict['loss_rpn_box_reg']+loss_dict['loss_classifier']
        losses_da=loss_dict['loss_da_instance']+loss_dict['loss_da_image']
        losses=[losses_det,losses_da]
        img_acc=model.da_heads.loss_evaluator.dis_img_result.clone()[1,:].mean(-1)
        ins_acc=model.da_heads.loss_evaluator.dis_ins_result.clone()[1,:].mean(-1)
        scale = 0.5 * (img_acc + ins_acc)
        scale = torch.nan_to_num(scale, nan=0.0).item()
        # reduce losses over all GPUs for logging purposes 
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        optimizer_new.DAGrad_backward(losses,scale)
        optimizer.step()

        with torch.no_grad():
            model(images, targets)

        if iteration>cfg.SOLVER.WARMUP_ITERS: 
            raw_dis_img_result=model.da_heads.loss_evaluator.dis_img_result.clone()
            raw_dis_ins_result=model.da_heads.loss_evaluator.dis_ins_result.clone()
            # img_result = raw_dis_img_result[1, :]
            # ins_result = raw_dis_ins_result[1, :]
            # is_fed_img = torch.all(torch.isfinite(img_result)) and torch.all(img_result < 0.5)
            # is_fed_ins = torch.all(torch.isfinite(ins_result)) and torch.all(ins_result < 0.5)
            img_loss_result = raw_dis_img_result[0, :]
            img_acc_result = raw_dis_img_result[1, :]
            ins_loss_result = raw_dis_ins_result[0, :]
            ins_acc_result = raw_dis_ins_result[1, :]
            is_fed_img = (
                torch.all(torch.isfinite(img_loss_result))
                and torch.all(torch.isfinite(img_acc_result))
                and torch.all(img_acc_result < 0.5)
            )
            is_fed_ins = (
                torch.all(torch.isfinite(ins_loss_result))
                and torch.all(torch.isfinite(ins_acc_result))
                and torch.all(ins_acc_result < 0.5)
            )
        else:
            is_fed_img=False
            is_fed_ins=False

        if  is_fed_img:
            # print("##################### conduct the FEDAVG in img level #####################")
            # print("ACC_IMG:",raw_dis_img_result[1,:],"STATE:",is_fed_img)
            weight_conv1=[]
            weight_conv2=[]
            for conv1_block, conv2_block in zip(model.da_heads.imghead.da_img_conv1_layers, model.da_heads.imghead.da_img_conv2_layers):
                weight_conv1.append(getattr(model.da_heads.imghead, conv1_block))
                weight_conv2.append(getattr(model.da_heads.imghead, conv2_block))
            weight_conv1_avg,weight_conv1_raw=update_discriminator_weights(weight_conv1,FedAvg)
            weight_conv2_avg,weight_conv2_raw=update_discriminator_weights(weight_conv2,FedAvg)

        if  is_fed_ins:
            # print("##################### conduct the FEDAVG in ins level #####################")
            # print("ACC_INS:",raw_dis_ins_result[1,:],"STATE:",is_fed_ins)
            ins_weight_fc1=[]
            ins_weight_fc2=[]
            ins_weight_fc3=[]
            for level, (fc1_da, fc2_da, fc3_da) in \
                enumerate(zip(model.da_heads.inshead.da_ins_fc1_layers,
                model.da_heads.inshead.da_ins_fc2_layers, 
                model.da_heads.inshead.da_ins_fc3_layers)):
                ins_weight_fc1.append(getattr(model.da_heads.inshead, fc1_da))
                ins_weight_fc2.append(getattr(model.da_heads.inshead, fc2_da))
                ins_weight_fc3.append(getattr(model.da_heads.inshead, fc3_da))
            weight_fc1_avg,weight_fc1_raw=update_discriminator_weights(ins_weight_fc1,FedAvg)
            weight_fc2_avg,weight_fc2_raw=update_discriminator_weights(ins_weight_fc2,FedAvg)
            weight_fc3_avg,weight_fc3_raw=update_discriminator_weights(ins_weight_fc3,FedAvg)

        if  is_fed_img or is_fed_ins:
            
            with torch.no_grad():
                model(images, targets)

        # if  is_fed_img:
        #     # print("##################### conduct the ADA in img level #####################")
        #     after_dis_img_result=model.da_heads.loss_evaluator.dis_img_result.clone()
        #     norm_gap_list=after_dis_img_result-raw_dis_img_result
        #     # print("img level acc and loss gap: ", norm_gap_list)
        #     value_list_weight=-norm_gap_list[0,:]
        #     img_weight= torch.softmax(value_list_weight,dim=0)
        #     # print("img level fed personal weight: (img_weight,img_bias)", img_weight)
        #     load_discriminator_with_weights(weight_conv1,weight_conv1_avg,img_weight,weight_conv1_raw)
        #     load_discriminator_with_weights(weight_conv2,weight_conv2_avg,img_weight,weight_conv2_raw)
        
        # if  is_fed_ins:
        #     # print("##################### conduct the ADA in ins level #####################")
        #     after_dis_ins_result=model.da_heads.loss_evaluator.dis_ins_result.clone()
        #     norm_gap_list_ins=after_dis_ins_result-raw_dis_ins_result
        #     # print("ins level acc and loss gap: ", norm_gap_list_ins)
        #     value_ins_weight=-norm_gap_list_ins[0,:]
        #     ins_weight= torch.softmax(value_ins_weight,dim=0)
        #     # print("ins level fed personal weight: (ins_weight,ins_bias)", ins_weight)
        #     load_discriminator_with_weights(ins_weight_fc1,weight_fc1_avg,ins_weight,weight_fc1_raw)
        #     load_discriminator_with_weights(ins_weight_fc2,weight_fc2_avg,ins_weight,weight_fc2_raw)
        #     load_discriminator_with_weights(ins_weight_fc3,weight_fc3_avg,ins_weight,weight_fc3_raw)
        if  is_fed_img:
            # print("##################### conduct the ADA in img level #####################")
            after_dis_img_result=model.da_heads.loss_evaluator.dis_img_result.clone()
            index_nan_list_img=~torch.isfinite(after_dis_img_result)
            after_dis_img_result = torch.nan_to_num(after_dis_img_result, nan=-1, posinf=0.0, neginf=0.0)
            # raw_dis_img_result = torch.nan_to_num(raw_dis_img_result, nan=-1, posinf=0.0, neginf=0.0)
            norm_gap_list=after_dis_img_result-raw_dis_img_result
            value_list_weight=-norm_gap_list[0,:]
            value_list_weight=value_list_weight.masked_fill(index_nan_list_img.any(dim=0), float("-inf"))
            if torch.isinf(value_list_weight).all():
                img_weight=torch.ones_like(value_list_weight)/value_list_weight.numel()
            else:
                img_weight= torch.softmax(value_list_weight,dim=0)
            print("img level fed personal weight: (img_weight,img_bias)", img_weight)
            load_discriminator_with_weights(weight_conv1,weight_conv1_avg,img_weight,weight_conv1_raw)
            load_discriminator_with_weights(weight_conv2,weight_conv2_avg,img_weight,weight_conv2_raw)
            if (~torch.isfinite(after_dis_img_result)).any().item():
                for i in torch.where(index_nan_list_img.any(dim=0))[0].tolist():
                    weight_conv1[i].load_state_dict(weight_conv1_raw[i])
                    weight_conv2[i].load_state_dict(weight_conv2_raw[i])
            else:
                fed_iter+=1 
               
        
        if  is_fed_ins:
            # print("##################### conduct the ADA in ins level #####################")
            after_dis_ins_result=model.da_heads.loss_evaluator.dis_ins_result.clone()
            index_nan_list_ins=~torch.isfinite(after_dis_ins_result)
            after_dis_ins_result = torch.nan_to_num(after_dis_ins_result, nan=-1, posinf=0.0, neginf=0.0)
            # raw_dis_ins_result = torch.nan_to_num(raw_dis_ins_result, nan=-1, posinf=0.0, neginf=0.0)
            norm_gap_list_ins=after_dis_ins_result-raw_dis_ins_result
            value_ins_weight=-norm_gap_list_ins[0,:]
            value_ins_weight=value_ins_weight.masked_fill(index_nan_list_ins.any(dim=0), float("-inf"))
            if torch.isinf(value_ins_weight).all():
                ins_weight=torch.ones_like(value_ins_weight)/value_ins_weight.numel()
            else:
                ins_weight= torch.softmax(value_ins_weight,dim=0)
            load_discriminator_with_weights(ins_weight_fc1,weight_fc1_avg,ins_weight,weight_fc1_raw)
            load_discriminator_with_weights(ins_weight_fc2,weight_fc2_avg,ins_weight,weight_fc2_raw)
            load_discriminator_with_weights(ins_weight_fc3,weight_fc3_avg,ins_weight,weight_fc3_raw)
            # for i in torch.where(index_nan_list_ins.any(dim=0))[0].tolist():
            # for i in torch.where(index_nan_list_ins.any(dim=0))[0].tolist():
            #     print(index_nan_list_ins)
            #     ins_weight_fc1[i].load_state_dict(weight_fc1_raw[i])
            #     ins_weight_fc2[i].load_state_dict(weight_fc2_raw[i])
            #     ins_weight_fc3[i].load_state_dict(weight_fc3_raw[i])
            if (~torch.isfinite(after_dis_ins_result)).any().item():
                for i in range(len(norm_gap_list_ins[0,:])):
                    ins_weight_fc1[i].load_state_dict(weight_fc1_raw[i])
                    ins_weight_fc2[i].load_state_dict(weight_fc2_raw[i])
                    ins_weight_fc3[i].load_state_dict(weight_fc3_raw[i])
            else:
                fed_iter+=1 

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
    
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            logger.info("fed_iter: {}/{}".format(fed_iter, iteration))
        if iteration % checkpoint_period == 0 and iteration != 0 and iteration>=min_val_step:
            model.eval()
            val_results = validataion(cfg, model, val_data_loader[0], distributed=False)

            if type(val_results)== dict:
                AP50_online = val_results['map'] * 100
                meters.update(AP50=AP50_online)
            else:
                val_results = val_results[0]
                # used for saving model
                AP50_online = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                # used for logging
                meter_AP50= val_results.results['bbox']['AP50'] * 100
                meter_AP = val_results.results['bbox']['AP']* 100
                meters.update(AP50=meter_AP50, AP =meter_AP )

            if AP50_online > max_AP50_bank:
                max_AP50_bank = AP50_online
                
                checkpointer.save("best_model_updated_{}_{:07d}".format(AP50_online, iteration), **arguments)
            logger.info('best mAP: {}'.format(max_AP50_bank))
            model.train()
        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )



def validataion(cfg, model, data_loader, distributed=False):
    iou_types = ("bbox",)
    # iou_types = ("map",)
    dataset_name = cfg.DATASETS.TEST

    results = inference(
        model,
        data_loader,
        dataset_name=dataset_name,
        iou_types=iou_types,
        box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=None,
    )
    # synchronize()
    return results