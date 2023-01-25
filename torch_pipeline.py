#!/usr/bin/python3

import os
import sys
import torch
import h5py
import logging
import traceback
import hdf5storage
import numpy as np


from absl import app
from tqdm import tqdm
from glob import glob
from absl import flags
from scipy import io as sio
from torch_dataLoader import get_dataset
from torch.utils import data as torchdata
from torch_models import LinVARX, DeepVARX
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(filename="DeepVARX.log", level=logging.DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_integer('lags', 60, """Number of lags""")

flags.DEFINE_integer('gpu', 0,
						"""GPU Selection.""")
flags.DEFINE_integer('epochs', 2,
						"""Number of epochs.""")
flags.DEFINE_integer('batch_size', 32, """Batch size.""")
flags.DEFINE_integer('stepSize', 1,
						"""Number of step size.""")
flags.DEFINE_string('suffix', '',
						"""Save file suffix""")
flags.DEFINE_string('logdir', None,
						"""Log dir""")
flags.DEFINE_string('config_file', None,
						"""Config file""")
flags.DEFINE_integer('predictionHorizon', 1,
						"""Steps ahead to predict.""")
flags.DEFINE_string('input', "/path/to/folder/pipeline/prepped", 
						"""Input location of pipeline results.""")
flags.DEFINE_string('output', "/path/to/folder/pipeline/results", 
						"""Output location of pipeline results.""")


def singleBatchTraining(model, data, target, history=False):
	model.optimiser.zero_grad()
	prediction = model(data)
	loss = model.loss_fn(prediction, target)
	model.metrics.update((prediction, target))
	loss.backward()
	model.optimiser.step()
	
	if history:
		return loss.item()


def singleBatchEvaluation(model, data, target):
	model.eval()
	with torch.no_grad():
		prediction = model(data)
		loss = model.loss_fn(prediction, target)
		model.metrics.update((prediction, target))
	return loss.item()


def singleBatchPrediction(model, data):
	model.eval()
	with torch.no_grad():
		return model(data)


def resetParameter(model):
	for _, module in model.named_children():
		try:
			module.reset_parameters()
		except AttributeError:
			pass

def reset_parameters(model):
    for _, module in model.named_modules():
        try:
            module.reset_parameters()
        except AttributeError:
            pass



########## START OF THE MAIN BLOCK #############

def main(argv):
	if FLAGS.config_file is not None:
		with open(FLAGS.config_file, "r") as _file:
			config = yaml.safe_load(_file)

		for key in config.keys():
			FLAGS.__dict__[key] = config[key]
	# SELECT GPU
	suffix = "Lags{}StepSize{}PredictionHorizon{}".format(FLAGS.lags,FLAGS.stepSize,FLAGS.predictionHorizon)
	print(suffix)
	# # LOAD DATA

	suffix+=FLAGS.suffix

	modelHistory = h5py.File(os.path.join(FLAGS.output, 'modelHistory{}{}{}.h5'.format(suffix, 'Epochs', str(FLAGS.epochs))), "w")
	OLEvaluation = h5py.File(os.path.join(FLAGS.output, 'OLEvaluation{}{}{}.h5'.format(suffix, 'Epochs', str(FLAGS.epochs))), "w")
	OLPrediction = h5py.File(os.path.join(FLAGS.output, 'OLPrediction{}{}{}.h5'.format(suffix, 'Epochs', str(FLAGS.epochs))), "w")
	CLEvaluation = h5py.File(os.path.join(FLAGS.output, 'CLEvaluation{}{}{}.h5'.format(suffix, 'Epochs', str(FLAGS.epochs))), "w")
	CLPrediction = h5py.File(os.path.join(FLAGS.output, 'CLPrediction{}{}{}.h5'.format(suffix, 'Epochs', str(FLAGS.epochs))), "w")

	logdir = FLAGS.logdir if FLAGS.logdir is not None else os.path.join(
		os.path.dirname(__file__),
		"runs"
	)
	
	try:
		device = f'cuda:{FLAGS.gpu}'
		linVARX = LinVARX(lags=FLAGS.lags).to(device)
		justGaze = DeepVARX(gazeOnly=True, lags=FLAGS.lags).to(device)
		jointAngleGazewithHead = DeepVARX(exoInput=3, lags=FLAGS.lags).to(device)
		jointAngleGazewithHands = DeepVARX(exoInput=6, lags=FLAGS.lags).to(device)
		jointAngleGazewithBody = DeepVARX(exoInput=66, lags=FLAGS.lags).to(device)
		angularVelocityGazewithHead = DeepVARX(exoInput=3, lags=FLAGS.lags).to(device)
		angularVelocityGazewithBody = DeepVARX(exoInput=69, lags=FLAGS.lags).to(device)

		for mode in ['in_subject', 'cross_subject']:
			block_set = get_dataset(FLAGS.input, suffix, 600, 600, mode=mode, batch_size=128)
			for idx, (subject, block_data) in enumerate(tqdm(block_set)):
				for block_idx, (train_data, test_data, closed_loop_block) in enumerate(block_data):
					writer = SummaryWriter(
						log_dir=os.path.join(
							logdir,
							subject
						)
					)
					print('PROCESSING {}'.format(subject))
					logging.info("PROCESSING {}".format(subject))
						
					trainData = torchdata.DataLoader(
						train_data,
						batch_size=FLAGS.batch_size,
						shuffle=True, 
						pin_memory=True
					)
					testData = torchdata.DataLoader(
						test_data,
						shuffle=False,
						batch_size=FLAGS.batch_size,
						pin_memory=True
					)


					print("CONSTRUCTING MODEL...")
					logging.info("CONSTRUCTING MODEL")
				
					reset_parameters(linVARX)
					reset_parameters(justGaze)
					reset_parameters(jointAngleGazewithHead)
					reset_parameters(jointAngleGazewithHands)
					reset_parameters(jointAngleGazewithBody)
					reset_parameters(angularVelocityGazewithHead)
					reset_parameters(angularVelocityGazewithBody)



					print("FITTING...")
					logging.info("FITTING...")
					
					history = {
								'linVARX':{'losses': [],
										'metrics': [],
										},
								'G': {'losses': [],
										'metrics': [],
										},
								'JAGwH': {'losses': [],
										'metrics': [],
										},
								'JAGwHn': {'losses': [],
										'metrics': [],
										},
								'JAGwB': {'losses': [],
										'metrics': [],
										},
								'aVGwH': {'losses': [],
										'metrics': [],
										},
								'aVGwB': {'losses': [],
										'metrics': [],
										},
							}
					
					postfixes = {
								'linVARX':{'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
								'G': {'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
								'JAGwH': {'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
								'JAGwHn': {'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
								'JAGwB': {'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
								'aVGwH': {'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
								'aVGwB': {'losses': 0,
										'metrics': 0,
										'val_losses': 0,
										'val_metrics': 0
										},
					}

					train_count = 0
					test_count = 0
					val_count = 0
					for epoch in range(FLAGS.epochs):
						
						linVARX.train()
						justGaze.train() 
						jointAngleGazewithHead.train()
						jointAngleGazewithHands.train()
						jointAngleGazewithBody.train()
						angularVelocityGazewithHead.train()
						angularVelocityGazewithBody.train() 

						linVARX.metrics.reset()
						justGaze.metrics.reset()
						jointAngleGazewithHead.metrics.reset()
						jointAngleGazewithHands.metrics.reset()
						jointAngleGazewithBody.metrics.reset()
						angularVelocityGazewithHead.metrics.reset()
						angularVelocityGazewithBody.metrics.reset()


						trainLoop = tqdm(trainData, desc=f"{epoch + 1}/{FLAGS.epochs}", position=0, postfix=postfixes)
						for idx, (data, target) in enumerate(trainLoop):
							target = target.cuda(device, non_blocking=True)
							eyeinput, jointAngle, angularVelocity, noise = data 
							eyeinput = eyeinput.cuda(device, non_blocking=True)
							jointAngle = jointAngle.cuda(device, non_blocking=True)
							angularVelocity = angularVelocity.cuda(device, non_blocking=True)
							noise = noise.cuda(device, non_blocking=True)
							
							linVARXhistory = singleBatchTraining(linVARX, (eyeinput, jointAngle), target, history=True)
							justGazehistory = singleBatchTraining(justGaze, eyeinput, target, history=True)
							jointAngleGazewithHeadhistory = singleBatchTraining(jointAngleGazewithHead, (eyeinput, jointAngle[:,:,15:18]), target, history=True)
							jointAngleGazewithHandshistory = singleBatchTraining(jointAngleGazewithHands, (eyeinput, jointAngle[:,:,[27,28,29,39,40,41]]), target, history=True)
							jointAngleGazewithBodyhistory = singleBatchTraining(jointAngleGazewithBody, (eyeinput, jointAngle), target, history=True)
							angularVelocityGazewithHeadhistory = singleBatchTraining(angularVelocityGazewithHead, (eyeinput, angularVelocity[:,:,18:21]), target, history=True)
							angularVelocityGazewithBodyhistory = singleBatchTraining(angularVelocityGazewithBody, (eyeinput, angularVelocity), target, history=True)

							linVARX_metrics = linVARX.metrics.compute()
							G_metrics = justGaze.metrics.compute()
							JAGwH_metrics = jointAngleGazewithHead.metrics.compute()
							JAGwHn_metrics = jointAngleGazewithHands.metrics.compute()
							JAGwB_metrics = jointAngleGazewithBody.metrics.compute()
							aVGwH_metrics = angularVelocityGazewithHead.metrics.compute()
							aVGwB_metrics = angularVelocityGazewithBody.metrics.compute()

							if train_count % 10 == 0:

								# Losses
								writer.add_scalar("linVARX/training/losses", linVARXhistory, train_count)
								writer.add_scalar("G/training/losses", justGazehistory, train_count)
								writer.add_scalar("JAGwH/training/losses", jointAngleGazewithHeadhistory, train_count)
								writer.add_scalar("JAGwHn/training/losses", jointAngleGazewithHandshistory, train_count)
								writer.add_scalar("JAGwB/training/losses", jointAngleGazewithBodyhistory, train_count)
								writer.add_scalar("aVGwH/training/losses", angularVelocityGazewithHeadhistory, train_count)
								writer.add_scalar("aVGwB/training/losses", angularVelocityGazewithBodyhistory, train_count)

								# Metrics
								writer.add_scalar("linVARX/training/metrics", linVARX_metrics, train_count)
								writer.add_scalar("G/training/metrics", G_metrics, train_count)
								writer.add_scalar("JAGwH/training/metrics", JAGwH_metrics, train_count)
								writer.add_scalar("JAGwHn/training/metrics", JAGwHn_metrics, train_count)
								writer.add_scalar("JAGwB/training/metrics", JAGwB_metrics, train_count)
								writer.add_scalar("aVGwH/training/metrics", aVGwH_metrics, train_count)
								writer.add_scalar("aVGwB/training/metrics", aVGwB_metrics, train_count)
		                                        					
								writer.flush()

							postfixes["linVARX"]["losses"] = f"{linVARXhistory:.4f}"
							postfixes["G"]["losses"] = f"{justGazehistory:.4f}"
							postfixes["JAGwH"]["losses"] = f"{jointAngleGazewithHeadhistory:.4f}"
							postfixes["JAGwHn"]["losses"] = f"{jointAngleGazewithHandshistory:.4f}"
							postfixes["JAGwB"]["losses"] = f"{jointAngleGazewithBodyhistory:.4f}"
							postfixes["aVGwH"]["losses"] = f"{angularVelocityGazewithHeadhistory:.4f}"
							postfixes["aVGwB"]["losses"] = f"{angularVelocityGazewithBodyhistory:.4f}"

							postfixes["linVARX"]["metrics"] = f"{linVARX_metrics:.4f}"
							postfixes["G"]["metrics"] = f"{G_metrics:.4f}"
							postfixes["JAGwH"]["metrics"] = f"{JAGwH_metrics:.4f}"
							postfixes["JAGwHn"]["metrics"] = f"{JAGwHn_metrics:.4f}"
							postfixes["JAGwB"]["metrics"] = f"{JAGwB_metrics:.4f}"
							postfixes["aVGwH"]["metrics"] = f"{aVGwH_metrics:.4f}"
							postfixes["aVGwB"]["metrics"] = f"{aVGwB_metrics:.4f}"
       
							history["linVARX"]["losses"].append(linVARXhistory)
							history["G"]["losses"].append(justGazehistory)
							history["JAGwH"]["losses"].append(jointAngleGazewithHeadhistory)
							history["JAGwHn"]["losses"].append(jointAngleGazewithHandshistory)
							history["JAGwB"]["losses"].append(jointAngleGazewithBodyhistory)
							history["aVGwH"]["losses"].append(angularVelocityGazewithHeadhistory)
							history["aVGwB"]["losses"].append(angularVelocityGazewithBodyhistory)

							train_count += 1

							trainLoop.set_postfix(postfixes)


						history["linVARX"]["metrics"].append(linVARX.metrics.compute())
						history["G"]["metrics"].append(justGaze.metrics.compute())
						history["JAGwH"]["metrics"].append(jointAngleGazewithHead.metrics.compute())
						history["JAGwHn"]["metrics"].append(jointAngleGazewithHands.metrics.compute())
						history["JAGwB"]["metrics"].append(jointAngleGazewithBody.metrics.compute())
						history["aVGwH"]["metrics"].append(angularVelocityGazewithHead.metrics.compute())
						history["aVGwB"]["metrics"].append(angularVelocityGazewithBody.metrics.compute())


						linVARX.metrics.reset()
						justGaze.metrics.reset()
						jointAngleGazewithHead.metrics.reset()
						jointAngleGazewithHands.metrics.reset()
						jointAngleGazewithBody.metrics.reset()
						angularVelocityGazewithHead.metrics.reset()
						angularVelocityGazewithBody.metrics.reset()


						for idx, (data, target) in enumerate(testData):
							target = target.cuda(device, non_blocking=True)
							eyeinput, jointAngle, angularVelocity, noise = data
							eyeinput = eyeinput.cuda(device, non_blocking=True)
							jointAngle = jointAngle.cuda(device, non_blocking=True)
							angularVelocity = angularVelocity.cuda(device, non_blocking=True)
							noise = noise.cuda(device, non_blocking=True)
							
							linVARXevaluation = singleBatchEvaluation(linVARX, (eyeinput, jointAngle), target)
							justGazeevaluation = singleBatchEvaluation(justGaze, eyeinput, target)
							jointAngleGazewithHeadevaluation = singleBatchEvaluation(jointAngleGazewithHead, (eyeinput, jointAngle[:,:,15:18]), target)
							jointAngleGazewithHandsevaluation = singleBatchEvaluation(jointAngleGazewithHands, (eyeinput, jointAngle[:,:,[27,28,29,39,40,41]]), target)
							jointAngleGazewithBodyevaluation = singleBatchEvaluation(jointAngleGazewithBody, (eyeinput, jointAngle), target)
							angularVelocityGazewithHeadevaluation = singleBatchEvaluation(angularVelocityGazewithHead, (eyeinput, angularVelocity[:,:,18:21]), target)
							angularVelocityGazewithBodyevaluation = singleBatchEvaluation(angularVelocityGazewithBody, (eyeinput, angularVelocity), target)

							linVARX_metrics = linVARX.metrics.compute()
							G_metrics = justGaze.metrics.compute()
							JAGwH_metrics = jointAngleGazewithHead.metrics.compute()
							JAGwHn_metrics = jointAngleGazewithHands.metrics.compute()
							JAGwB_metrics = jointAngleGazewithBody.metrics.compute()
							aVGwH_metrics = angularVelocityGazewithHead.metrics.compute()
							aVGwB_metrics = angularVelocityGazewithBody.metrics.compute()

							if test_count % 10 == 0:


								writer.add_scalar("linVARX/testing/losses", linVARXevaluation, test_count)
								writer.add_scalar("G/testing/losses", justGazeevaluation, test_count)
								writer.add_scalar("JAGwH/testing/losses", jointAngleGazewithHeadevaluation, test_count)
								writer.add_scalar("JAGwHn/testing/losses", jointAngleGazewithHandsevaluation, test_count)
								writer.add_scalar("JAGwB/testing/losses", jointAngleGazewithBodyevaluation, test_count)
								writer.add_scalar("aVGwH/testing/losses", angularVelocityGazewithHeadevaluation, test_count)
								writer.add_scalar("aVGwB/testing/losses", angularVelocityGazewithBodyevaluation, test_count)


								writer.add_scalar("linVARX/testing/metrics", linVARX_metrics, test_count)
								writer.add_scalar("G/testing/metrics", G_metrics, test_count)
								writer.add_scalar("JAGwH/testing/metrics", JAGwH_metrics, test_count)
								writer.add_scalar("JAGwHn/testing/metrics", JAGwHn_metrics, test_count)
								writer.add_scalar("JAGwB/testing/metrics", JAGwB_metrics, test_count)
								writer.add_scalar("aVGwH/testing/metrics", aVGwH_metrics, test_count)
								writer.add_scalar("aVGwB/testing/metrics", aVGwB_metrics, test_count)

								writer.flush()


							postfixes["linVARX"]["val_losses"] = f"{linVARXevaluation:.4f}"
							postfixes["G"]["val_losses"] = f"{justGazeevaluation:.4f}"
							postfixes["JAGwH"]["val_losses"] = f"{jointAngleGazewithHeadevaluation:.4f}"
							postfixes["JAGwHn"]["val_losses"] = f"{jointAngleGazewithHandsevaluation:.4f}"
							postfixes["JAGwB"]["val_losses"] = f"{jointAngleGazewithBodyevaluation:.4f}"
							postfixes["aVGwH"]["val_losses"] = f"{angularVelocityGazewithHeadevaluation:.4f}"
							postfixes["aVGwB"]["val_losses"] = f"{angularVelocityGazewithBodyevaluation:.4f}"

							postfixes["linVARX"]["val_metrics"] = f"{linVARX_metrics:.4f}"
							postfixes["G"]["val_metrics"] = f"{G_metrics:.4f}"
							postfixes["JAGwH"]["val_metrics"] = f"{JAGwH_metrics:.4f}"
							postfixes["JAGwHn"]["val_metrics"] = f"{JAGwHn_metrics:.4f}"
							postfixes["JAGwB"]["val_metrics"] = f"{JAGwB_metrics:.4f}"
							postfixes["aVGwH"]["val_metrics"] = f"{aVGwH_metrics:.4f}"
							postfixes["aVGwB"]["val_metrics"] = f"{aVGwB_metrics:.4f}"

							test_count += 1

							trainLoop.set_postfix(postfixes) 

					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX", "metrics"]), data=np.asanyarray(history["linVARX"]["metrics"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G", "metrics"]), data=np.asanyarray(history["G"]["metrics"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH", "metrics"]), data=np.asanyarray(history["JAGwH"]["metrics"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn", "metrics"]), data=np.asanyarray(history["JAGwHn"]["metrics"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB", "metrics"]), data=np.asanyarray(history["JAGwB"]["metrics"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH", "metrics"]), data=np.asanyarray(history["aVGwH"]["metrics"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB", "metrics"]), data=np.asanyarray(history["aVGwB"]["metrics"]).squeeze())

					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX", "losses"]), data=np.asanyarray(history["linVARX"]["losses"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G", "losses"]), data=np.asanyarray(history["G"]["losses"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH", "losses"]), data=np.asanyarray(history["JAGwH"]["losses"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn", "losses"]), data=np.asanyarray(history["JAGwHn"]["losses"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB", "losses"]), data=np.asanyarray(history["JAGwB"]["losses"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH", "losses"]), data=np.asanyarray(history["aVGwH"]["losses"]).squeeze())
					modelHistory.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB", "losses"]), data=np.asanyarray(history["aVGwB"]["losses"]).squeeze())
					
					print("SAVING...")
					logging.info("SAVING...")
					

					torch.save({
						"LinVARX": {
							'model': linVARX.state_dict(),
							'optimiser': linVARX.optimiser.state_dict(),
						}, 
						"G": {
							'model': justGaze.state_dict(),
							'optimiser': justGaze.optimiser.state_dict(),
						}, 
						"JAGwH":  {
							'model': jointAngleGazewithHead.state_dict(),
							'optimiser': jointAngleGazewithHead.optimiser.state_dict(),
						}, 
						"JAGwHn":  {
							'model': jointAngleGazewithHands.state_dict(),
							'optimiser': jointAngleGazewithHands.optimiser.state_dict(),
						}, 
						"JAGwB":  {
							'model': jointAngleGazewithBody.state_dict(),
							'optimiser': jointAngleGazewithBody.optimiser.state_dict(),
						},
						"aVGwH":  {
							'model': angularVelocityGazewithHead.state_dict(),
							'optimiser': angularVelocityGazewithHead.optimiser.state_dict(),
						},  
						"aVGwB":  {
							'model': angularVelocityGazewithBody.state_dict(),
							'optimiser': angularVelocityGazewithBody.optimiser.state_dict(),
						},

					}, 'checkpoint.pt'
					)
					
					print("EVALUATING...")
					logging.info("EVALUATING...")

					
					history = {
								'linVARX':{'val_losses': [],
										'val_metrics': [],
										},
								'G': {'val_losses': [],
										'val_metrics': [],
										},
								'JAGwH': {'val_losses': [],
										'val_metrics': [],
										},
								'JAGwHn': {'val_losses': [],
										'val_metrics': [],
										},
								'JAGwB': {'val_losses': [],
										'val_metrics': [],
										},
								'aVGwH': {'val_losses': [],
										'val_metrics': [],
										},
								'aVGwB': {'val_losses': [],
										'val_metrics': [],
										},
							}


					angularVelocityGazewithHeadOLprediction = [] 
					angularVelocityGazewithBodyOLprediction = [] 

					jointAngleGazewithHeadOLprediction = []
					jointAngleGazewithHandsOLprediction = []
					jointAngleGazewithBodyOLprediction = []
					justGazeOLprediction = []
					linVARXOLprediction = []


					linVARX.metrics.reset()
					justGaze.metrics.reset()
					jointAngleGazewithHead.metrics.reset()
					jointAngleGazewithHands.metrics.reset()
					jointAngleGazewithBody.metrics.reset()
					angularVelocityGazewithHead.metrics.reset()
					angularVelocityGazewithBody.metrics.reset()

					for (data, target) in tqdm(testData, desc='Evaluating...'):
						target = target.cuda(device, non_blocking=True)
						eyeinput, jointAngle, angularVelocity, noise = data
						eyeinput = eyeinput.cuda(device, non_blocking=True)
						jointAngle = jointAngle.cuda(device, non_blocking=True)
						angularVelocity = angularVelocity.cuda(device, non_blocking=True)
						noise = noise.cuda(device, non_blocking=True)
						
						linVARXevaluation = singleBatchEvaluation(linVARX, (eyeinput, jointAngle), target)
						linVARXOLprediction.append(singleBatchPrediction(linVARX, (eyeinput, jointAngle)))

						justGazeevaluation = singleBatchEvaluation(justGaze, eyeinput, target)
						justGazeOLprediction.append(singleBatchPrediction(justGaze, eyeinput))

						jointAngleGazewithHeadevaluation = singleBatchEvaluation(jointAngleGazewithHead, (eyeinput, jointAngle[:,:,15:18]), target)
						jointAngleGazewithHeadOLprediction.append(singleBatchPrediction(jointAngleGazewithHead, (eyeinput, jointAngle[:,:,15:18])))
						
						jointAngleGazewithHandsevaluation = singleBatchEvaluation(jointAngleGazewithHands, (eyeinput, jointAngle[:,:,[27,28,29,39,40,41]]), target) #variables chosen to fit existing MVNX structure, may have to modify in future for alternative feature layouts
						jointAngleGazewithHandsOLprediction.append(singleBatchPrediction(jointAngleGazewithHands, (eyeinput, jointAngle[:,:,[27,28,29,39,40,41]])))

						jointAngleGazewithBodyevaluation = singleBatchEvaluation(jointAngleGazewithBody, (eyeinput, jointAngle), target)
						jointAngleGazewithBodyOLprediction.append(singleBatchPrediction(jointAngleGazewithBody, (eyeinput, jointAngle)))

						angularVelocityGazewithHeadevaluation = singleBatchEvaluation(angularVelocityGazewithHead, (eyeinput, angularVelocity[:,:,18:21]), target)
						angularVelocityGazewithHeadOLprediction.append(singleBatchPrediction(angularVelocityGazewithHead, (eyeinput, angularVelocity[:,:,18:21])))

						angularVelocityGazewithBodyevaluation = singleBatchEvaluation(angularVelocityGazewithBody, (eyeinput, angularVelocity), target)
						angularVelocityGazewithBodyOLprediction.append(singleBatchPrediction(angularVelocityGazewithBody, (eyeinput, angularVelocity)))


						history["linVARX"]["val_losses"].append(linVARXevaluation)
						history["G"]["val_losses"].append(justGazeevaluation)
						history["JAGwH"]["val_losses"].append(jointAngleGazewithHeadevaluation)
						history["JAGwHn"]["val_losses"].append(jointAngleGazewithHandsevaluation)
						history["JAGwB"]["val_losses"].append(jointAngleGazewithBodyevaluation)
						history["aVGwH"]["val_losses"].append(angularVelocityGazewithHeadevaluation)
						history["aVGwB"]["val_losses"].append(angularVelocityGazewithBodyevaluation)

						history["linVARX"]["val_metrics"].append(linVARX.metrics.compute())
						history["G"]["val_metrics"].append(justGaze.metrics.compute())
						history["JAGwH"]["val_metrics"].append(jointAngleGazewithHead.metrics.compute())
						history["JAGwHn"]["val_metrics"].append(jointAngleGazewithHands.metrics.compute())
						history["JAGwB"]["val_metrics"].append(jointAngleGazewithBody.metrics.compute())
						history["aVGwH"]["val_metrics"].append(angularVelocityGazewithHead.metrics.compute())
						history["aVGwB"]["val_metrics"].append(angularVelocityGazewithBody.metrics.compute())
					

					linVARX.metrics.reset()
					justGaze.metrics.reset()
					jointAngleGazewithHead.metrics.reset()
					jointAngleGazewithHands.metrics.reset()
					jointAngleGazewithBody.metrics.reset()
					angularVelocityGazewithHead.metrics.reset()
					angularVelocityGazewithBody.metrics.reset()

					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX", "val_metrics"]), data=np.asanyarray(history["linVARX"]["val_metrics"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G", "val_metrics"]), data=np.asanyarray(history["G"]["val_metrics"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH", "val_metrics"]), data=np.asanyarray(history["JAGwH"]["val_metrics"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn", "val_metrics"]), data=np.asanyarray(history["JAGwHn"]["val_metrics"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB", "val_metrics"]), data=np.asanyarray(history["JAGwB"]["val_metrics"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH", "val_metrics"]), data=np.asanyarray(history["aVGwH"]["val_metrics"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB", "val_metrics"]), data=np.asanyarray(history["aVGwB"]["val_metrics"]).squeeze())

					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX", "val_losses"]), data=np.asanyarray(history["linVARX"]["val_losses"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G", "val_losses"]), data=np.asanyarray(history["G"]["val_losses"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH", "val_losses"]), data=np.asanyarray(history["JAGwH"]["val_losses"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn", "val_losses"]), data=np.asanyarray(history["JAGwHn"]["val_losses"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB", "val_losses"]), data=np.asanyarray(history["JAGwB"]["val_losses"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH", "val_losses"]), data=np.asanyarray(history["aVGwH"]["val_losses"]).squeeze())
					OLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB", "val_losses"]), data=np.asanyarray(history["aVGwB"]["val_losses"]).squeeze())
					


					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX"]), data=torch.cat(linVARXOLprediction).cpu().numpy())
					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G"]), data=torch.cat(justGazeOLprediction).cpu().numpy())
					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH"]), data=torch.cat(jointAngleGazewithHeadOLprediction).cpu().numpy())
					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn"]), data=torch.cat(jointAngleGazewithHandsOLprediction).cpu().numpy())
					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB"]), data=torch.cat(jointAngleGazewithBodyOLprediction).cpu().numpy())
					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH"]), data=torch.cat(angularVelocityGazewithHeadOLprediction).cpu().numpy())
					OLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB"]), data=torch.cat(angularVelocityGazewithBodyOLprediction).cpu().numpy())




					logging.info("Lists instantiated")

					print('Closed Loop Block Length: ', len(closed_loop_block))
					for loop_block_idx, closed_loop_data in enumerate(closed_loop_block): 
						
						(
							angularVelocityGazewithHeadCLpredictions,
							angularVelocityGazewithBodyCLpredictions,
							jointAngleGazewithHeadCLpredictions, 
							jointAngleGazewithHandsCLpredictions, 
							jointAngleGazewithBodyCLpredictions,
							justGazeCLpredictions,
							linVARXCLpredictions, 
							angularVelocityGazewithHeadCLEvaluations,
							angularVelocityGazewithBodyCLEvaluations,
							jointAngleGazewithHeadCLEvaluations,
							jointAngleGazewithHandsCLEvaluations,
							jointAngleGazewithBodyCLEvaluations,
							justGazeCLEvaluations, 
							linVARXCLEvaluations,
							tempTestTarget,
							tempTestWhite
						) = ([] for i in range(18))
						val_count = 0
						closedLoopData = torchdata.DataLoader(
							closed_loop_data,
							shuffle=False,
							batch_size=1,
							pin_memory=True
						)
						print('Closed Loop Data Length: ', len(closedLoopData))

						for idx, (data, target) in enumerate(tqdm(closedLoopData)):
						
							eyeinput, jointAngle, angularVelocity, noise = data

							tempTestTarget.append(target)
							tempTestWhite.append(noise)

							eyeinput = eyeinput.cuda(device, non_blocking=True)
							jointAngle = jointAngle.cuda(device, non_blocking=True)
							angularVelocity = angularVelocity.cuda(device, non_blocking=True)
							noise = noise.cuda(device, non_blocking=True)
							target = target.cuda(device, non_blocking=True)

							eyeinput = eyeinput.squeeze(0)
							jointAngle = jointAngle.squeeze(0)
							angularVelocity = angularVelocity.squeeze(0)
							noise = noise.squeeze(0)
							target = target.squeeze(0)

							if idx == 0: 
								closedloopinput = eyeinput
								print(closedloopinput.shape)
								print(FLAGS.predictionHorizon)

								linVARXCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)
								angularVelocityGazewithHeadCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)
								angularVelocityGazewithBodyCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)
								jointAngleGazewithHeadCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)
								jointAngleGazewithHandsCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)
								jointAngleGazewithBodyCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)
								justGazeCLprediction = closedloopinput[:, -FLAGS.predictionHorizon:, :].squeeze(1)

								angularVelocityGazewithHeadCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
								angularVelocityGazewithBodyCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
								jointAngleGazewithHeadCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
								jointAngleGazewithHandsCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
								jointAngleGazewithBodyCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
								justGazeCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
								linVARXCLpast = closedloopinput[:, :-FLAGS.predictionHorizon, :]
				
							angularVelocityGazewithHeadCLinput = torch.cat([angularVelocityGazewithHeadCLpast, angularVelocityGazewithHeadCLprediction.unsqueeze(1)], dim=1)
							angularVelocityGazewithBodyCLinput = torch.cat([angularVelocityGazewithBodyCLpast, angularVelocityGazewithBodyCLprediction.unsqueeze(1)], dim=1)
							jointAngleGazewithHeadCLinput = torch.cat([jointAngleGazewithHeadCLpast, jointAngleGazewithHeadCLprediction.unsqueeze(1)], dim=1)
							jointAngleGazewithHandsCLinput = torch.cat([jointAngleGazewithHandsCLpast, jointAngleGazewithHandsCLprediction.unsqueeze(1)], dim=1)
							jointAngleGazewithBodyCLinput = torch.cat([jointAngleGazewithBodyCLpast, jointAngleGazewithBodyCLprediction.unsqueeze(1)], dim=1)
							justGazeCLinput = torch.cat([justGazeCLpast, justGazeCLprediction.unsqueeze(1)], dim=1)
							linVARXCLinput = torch.cat([linVARXCLpast, linVARXCLprediction.unsqueeze(1)], dim=1)

							linVARXCLevaluation = singleBatchEvaluation(linVARX, (linVARXCLinput, jointAngle), target)
							linVARXCLprediction = singleBatchPrediction(linVARX, (linVARXCLinput, jointAngle))
							linVARXCLpredictions.append(linVARXCLprediction) #save predicted value to our list

							justGazeCLevaluation = singleBatchEvaluation(justGaze, justGazeCLinput, target)
							justGazeCLprediction = singleBatchPrediction(justGaze, justGazeCLinput)
							justGazeCLpredictions.append(justGazeCLprediction)

							jointAngleGazewithHeadCLevaluation = singleBatchEvaluation(jointAngleGazewithHead, (jointAngleGazewithHeadCLinput, jointAngle[:,:,15:18]), target)
							jointAngleGazewithHeadCLprediction = singleBatchPrediction(jointAngleGazewithHead, (jointAngleGazewithHeadCLinput, jointAngle[:,:,15:18]))
							jointAngleGazewithHeadCLpredictions.append(jointAngleGazewithHeadCLprediction)

							jointAngleGazewithHandsCLevaluation = singleBatchEvaluation(jointAngleGazewithHands, (jointAngleGazewithHandsCLinput, jointAngle[:,:,[27,28,29,39,40,41]]), target)
							jointAngleGazewithHandsCLprediction = singleBatchPrediction(jointAngleGazewithHands, (jointAngleGazewithHandsCLinput, jointAngle[:,:,[27,28,29,39,40,41]]))
							jointAngleGazewithHandsCLpredictions.append(jointAngleGazewithHandsCLprediction)

							jointAngleGazewithBodyCLevaluation = singleBatchEvaluation(jointAngleGazewithBody, (jointAngleGazewithBodyCLinput, jointAngle), target)
							jointAngleGazewithBodyCLprediction = singleBatchPrediction(jointAngleGazewithBody, (jointAngleGazewithBodyCLinput, jointAngle))
							jointAngleGazewithBodyCLpredictions.append(jointAngleGazewithBodyCLprediction)

							angularVelocityGazewithHeadCLevaluation = singleBatchEvaluation(angularVelocityGazewithHead, (angularVelocityGazewithHeadCLinput, angularVelocity[:,:,18:21]), target)
							angularVelocityGazewithHeadCLprediction = singleBatchPrediction(angularVelocityGazewithHead, (angularVelocityGazewithHeadCLinput, angularVelocity[:,:,18:21]))
							angularVelocityGazewithHeadCLpredictions.append(angularVelocityGazewithHeadCLprediction)

							angularVelocityGazewithBodyCLevaluation = singleBatchEvaluation(angularVelocityGazewithBody, (angularVelocityGazewithBodyCLinput, angularVelocity), target)
							angularVelocityGazewithBodyCLprediction = singleBatchPrediction(angularVelocityGazewithBody, (angularVelocityGazewithBodyCLinput, angularVelocity))
							angularVelocityGazewithBodyCLpredictions.append(angularVelocityGazewithBodyCLprediction)


							angularVelocityGazewithHeadCLpast = angularVelocityGazewithHeadCLinput[:, FLAGS.predictionHorizon:, :]
							angularVelocityGazewithBodyCLpast = angularVelocityGazewithBodyCLinput[:, FLAGS.predictionHorizon:, :]
							jointAngleGazewithHeadCLpast = jointAngleGazewithHeadCLinput[:, FLAGS.predictionHorizon:, :]
							jointAngleGazewithHandsCLpast = jointAngleGazewithHandsCLinput[:, FLAGS.predictionHorizon:, :]
							jointAngleGazewithBodyCLpast = jointAngleGazewithBodyCLinput[:, FLAGS.predictionHorizon:, :]
							justGazeCLpast = justGazeCLinput[:, FLAGS.predictionHorizon:, :]
							linVARXCLpast = linVARXCLinput[:, FLAGS.predictionHorizon:, :]
						
							angularVelocityGazewithHeadCLEvaluations.append(angularVelocityGazewithHeadCLevaluation)
							angularVelocityGazewithBodyCLEvaluations.append(angularVelocityGazewithBodyCLevaluation)
							jointAngleGazewithHeadCLEvaluations.append(jointAngleGazewithHeadCLevaluation)
							jointAngleGazewithHandsCLEvaluations.append(jointAngleGazewithHandsCLevaluation)
							jointAngleGazewithBodyCLEvaluations.append(jointAngleGazewithBodyCLevaluation)
							justGazeCLEvaluations.append(justGazeCLevaluation)
							linVARXCLEvaluations.append(linVARXCLevaluation)

							linVARX_metrics = linVARX.metrics.compute()
							G_metrics = justGaze.metrics.compute()
							JAGwH_metrics = jointAngleGazewithHead.metrics.compute()
							JAGwHn_metrics = jointAngleGazewithHands.metrics.compute()
							JAGwB_metrics = jointAngleGazewithBody.metrics.compute()
							aVGwH_metrics = angularVelocityGazewithHead.metrics.compute()
							aVGwB_metrics = angularVelocityGazewithBody.metrics.compute()

						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "closedloopinput", f"closed_loop_block_{loop_block_idx}"]), data=closedloopinput.cpu().numpy())
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "testTarget", f"closed_loop_block_{loop_block_idx}"]), data=torch.stack(tempTestTarget).cpu().numpy())
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "testwhiteNoise", f"closed_loop_block_{loop_block_idx}"]), data=torch.stack(tempTestWhite).cpu().numpy())

						angularVelocityGazewithHeadCLpredictions = torch.cat(angularVelocityGazewithHeadCLpredictions).cpu().numpy()
						angularVelocityGazewithBodyCLpredictions = torch.cat(angularVelocityGazewithBodyCLpredictions).cpu().numpy()
						jointAngleGazewithHeadCLpredictions = torch.cat(jointAngleGazewithHeadCLpredictions).cpu().numpy()
						jointAngleGazewithHandsCLpredictions = torch.cat(jointAngleGazewithHandsCLpredictions).cpu().numpy()
						jointAngleGazewithBodyCLpredictions = torch.cat(jointAngleGazewithBodyCLpredictions).cpu().numpy()
						justGazeCLpredictions = torch.cat(justGazeCLpredictions).cpu().numpy()
						linVARXCLpredictions = torch.cat(linVARXCLpredictions).cpu().numpy()

						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(linVARXCLpredictions))
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(justGazeCLpredictions))
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(jointAngleGazewithHeadCLpredictions))
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(jointAngleGazewithHandsCLpredictions))
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(jointAngleGazewithBodyCLpredictions))
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(angularVelocityGazewithHeadCLpredictions))
						CLPrediction.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB", f"closed_loop_block_{loop_block_idx}"]), data=np.squeeze(angularVelocityGazewithBodyCLpredictions))

						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "linVARX", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(linVARXCLEvaluations))
						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "G", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(justGazeCLEvaluations))
						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwH", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(jointAngleGazewithHeadCLEvaluations))
						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwHn", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(jointAngleGazewithHandsCLEvaluations))
						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "JAGwB", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(jointAngleGazewithBodyCLEvaluations))
						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwH", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(angularVelocityGazewithHeadCLEvaluations))
						CLEvaluation.create_dataset(os.sep.join([mode, subject, f"block_{block_idx}", "aVGwB", f"closed_loop_block_{loop_block_idx}", "losses"]), data=np.asanyarray(angularVelocityGazewithBodyCLEvaluations))
						
					
					writer.flush()
					writer.close()


		if not os.path.exists(os.path.join(FLAGS.output)):
			os.makedirs(FLAGS.output)

	except Exception as e:
		print('Something broke!')
		print('Exception', e)
		print('Traceback:', traceback.format_exc())
		logging.error(traceback.format_exc())
		if not os.path.exists(os.path.join(FLAGS.output)):
			os.makedirs(FLAGS.output)

		try:
			sys.exit(0)
		except:
			os._exit(0)

	finally:
		modelHistory.close()
		OLEvaluation.close()
		OLPrediction.close()
		CLEvaluation.close()
		CLPrediction.close()
		print("\a")

if __name__ == '__main__':
	app.run(main)
