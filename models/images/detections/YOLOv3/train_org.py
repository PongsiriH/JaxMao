import jax
import jax.numpy as jnp
import numpy as np

from jaxmao import Bind
from model import load_model
from backbone.model import *
from loss import YOLOv3Loss
from dataset import YOLODataset
import jaxmao.nn.optimizers as optim
from utils import Results, yolo2xywh, plot_labels, LABELS_DICT_CATEGORIZED_GTSRB
import pickle
import config
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import cells_to_bboxes, plot_image, nms, mean_avg_precision
from pprint import pprint
import random

key = jax.random.key(42)

def initialize_training(params, states, model: YOLOBackboneResidual, learning_rate, loss_lambda):
	model.set_trainable(True)
	params, states = model.init_yolo_head(params, states, key)

	yolo_loss = YOLOv3Loss(**loss_lambda, iou_scale=False, ciou_loss="sum")
	optimizer = optim.Adam(params, learning_rate)

	return params, states, yolo_loss, optimizer

def train_model(params, states, optimizer, num_epochs=20, save_best_path=None):
	@jax.jit
	def train_step(images, labels, params, states, optimizer_states):
		def apply_loss(images, labels, params, states):
			predictions, states, _ = model.apply(images, params, states)
			total_loss = 0.0
			max_obj_scores = [jax.nn.sigmoid(prediction[..., 0].max()) for prediction in predictions]
			component_loss = {'bbox_loss': 0.0, 'obj_loss': 0.0, 'noobj_loss': 0.0, 'cls_loss': 0.0}
			for prediction, label, anchor in zip(predictions, labels, config.ANCHORS):
				loss_scale, comp_loss_scale = yolo_loss.calculate_loss(jnp.array(prediction), jnp.array(label), jnp.array(anchor)) 
				total_loss += loss_scale / prediction.shape[2] / prediction.shape[3]
				for key in comp_loss_scale:
					component_loss[key] += comp_loss_scale[key]
			return total_loss, (states, max_obj_scores, component_loss)
		
		(loss_value, (states, max_obj_scores, component_loss)), gradients = (jax.value_and_grad(apply_loss, argnums=2, has_aux=True))(images, labels, params, states)
		params, optimizer_states = jax.jit(optimizer.step)(params, gradients, optimizer_states)
		return loss_value, max_obj_scores, component_loss, params, states, optimizer_states

	# """Training loop"""
	mem_epochs = 10
	stuck_counter = 0
	loss_threshold = 0.001
	best_loss = np.inf
	losses_last_epochs = deque(maxlen=mem_epochs)
	c = 0
	for epoch in tqdm(range(num_epochs), desc="Epoch", position=1):
		train_dataset.update_image_size(epoch)
		print(f"epoch{epoch}.. image_size{train_dataset.image_size}:\n" )
		if epoch == 3:
			optimizer.states['lr'] = LR1
		if epoch > 5:
			optimizer.states['lr'] *= LR_DECAY
		total_losses = 0.0

		total_losses = 0.0
		for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc="Batch", position=2):
			if batch_idx == 500: break # take 500 step per epochs 
			images = jnp.array(np.array(images))
			labels = [label.numpy() for label in labels]

			loss_value, max_obj_scores, component_loss, params, states, optimizer.states = train_step(
				images, labels, params, states, optimizer.states
			)
			total_losses += loss_value

		# """Post-Epoch"""
		print('max_obj_scores:', max_obj_scores)
		avg_loss = total_losses / len(train_loader)
		print(f'epoch{epoch}: (avg_loss, {avg_loss}), {[(key, value.item()) for key, value in component_loss.items()]}')

		if any(np.isnan(value) for value in component_loss.values()):
			print(f"NaN detected in loss components at epoch {epoch}")
			best_loss = avg_loss
			return model, params, states, best_loss        

		c += 1
		# if avg_loss < best_loss and c > 0:
		if False:
			c = 0
			best_loss = avg_loss
   
			if save_best_path:
				with open(save_best_path, 'wb') as f:
					pickle.dump((model, params, states), f)

				with Bind(model, params, states) as ctx:
					predictions = ctx.predict(images, batch_size=4)
					mAP_score = ctx.evaluate_loader(train_loader, mean_avg_precision, max_batches=10)
					print('mAP:')
					pprint(mAP_score)
     
				conf_thresh = np.mean([jax.nn.sigmoid(predictions[i][..., 0]).max().item() for i in range(3)])
				pred_boxes: list = []
				gts_boxes: list = []
				for j in range(len(predictions[0])):
					for anchors, prediction, label in zip(config.ANCHORS, predictions, labels):
						gts_boxes.append(cells_to_bboxes(label, anchors, is_pred=False))
						pred_boxes.append(cells_to_bboxes(prediction, anchors, is_pred=True))
					gts_boxes = np.concatenate(gts_boxes, 1)
					pred_boxes = np.concatenate(pred_boxes, 1)
					box_true = nms(gts_boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
					box_pred = nms(pred_boxes, iou_threshold=config.NMS_IOU_THRESH, threshold=conf_thresh, box_format="midpoint")
		
					plt.subplot(1, 2, 1)
					plot_image(images[j], box_true)
					plt.subplot(1, 2, 2)
					plot_image(images[j], box_pred)
					plt.savefig(save_best_path.replace('.pkl', f'_{j}.jpg'), dpi=600)
					plt.close()
     
		losses_last_epochs.append(avg_loss)
		model.save_model(save_best_path, params, states)
		if epoch >= mem_epochs:
			mean_last_mem_epochs = sum(losses_last_epochs) / len(losses_last_epochs)
			if avg_loss < best_loss:
				best_loss = avg_loss
				stuck_counter = 0
			elif avg_loss > mean_last_mem_epochs: # 10 epochs
				print(f"Training stopped early at epoch {epoch}. Current loss is greater than the mean of the last {mem_epochs} epochs.")
				return model, params, states, best_loss
			elif abs(avg_loss - best_loss) < loss_threshold:
				stuck_counter += 1
				if stuck_counter >= 5:
					# Reduce learning rate
					optimizer.states['lr'] *= 0.7
					stuck_counter = 0  # Reset counter after reducing learning rate
					print(f"Reduced learning rate to {optimizer.states['lr']} at epoch {epoch}")
			else:
				stuck_counter = 0
	try:
		print(f'----------End Epoch {epoch} with avg_loss: {avg_loss} and mAP@50: {mAP_score["map_50"]}----------')
	except:
		print(f'----------End Epoch {epoch} with avg_loss: {avg_loss}----------')
	return model, params, states, best_loss

def generate_hyperparameters():
	lr0 = np.random.uniform(low=1e-10, high=1e-4)		
	lambdas = {
		'lambda_box': np.random.uniform(low=1.0, high=3.0),
		'lambda_obj': np.random.uniform(low=0.5, high=2.0),
		'lambda_noobj': np.random.uniform(low=0.005, high=2.0),
		'lambda_class': np.random.uniform(low=0.05, high=2.0),
	}
	lrs = {
		'lr0' : lr0,
		'lr1' : lr0*5,
		'weight_decay' : np.random.uniform(low=0.8, high=0.99)
	}
	return lambdas, lrs

def save_best_model(filename, model, params, states, best_params):
	with open(filename, 'wb') as f:
		pickle.dump({
			'model': model,
			'params': params,
			'states': states,
			'best_params': best_params
		}, f)
		
if __name__ == '__main__':
	image_sizes = [(224, 224), (256, 256), (320,320), (352, 352), (416, 416)]
	image_sizes = [(320,320),]
	NUM_CLASSES = 4
	# image_sizes = [(240, 240), (320, 320), (352, 352), (416, 416),]
	# image_sizes = [(240, 240), (320, 320), (352, 352), (416, 416), (480, 480), (640, 640)]
	random.shuffle(image_sizes)
	train_dataset = YOLODataset(
			img_dir='/home/jaxmao/dataset/Road Sign Dataset/images/train',
			image_sizes=image_sizes,
			change_size_interval=2,
			C=NUM_CLASSES,
			anchors=config.ANCHORS,
			builder_transform=config.build_train_transform,
			# cutmix=True,
			# builder_cutmix_transform=config.build_train_mosaic_transform,
			mosaic=0.6,
			builder_mosaic_transform=config.build_train_mosaic_transform
		)
	train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
 
	num_search_iterations: int = 1
	best_loss: float = np.inf
	best_params = None

	model: YOLOBackboneResidual
	
	for iteration in tqdm(range(5, 7), desc="Iteration", position=0):
		# model, params, states = load_model('YOLOv3_3/backbone/results/yolov3_backbone.pkl')
		model, params, states = load_model('YOLOv3_ciou_greate.pkl')
		# model.num_classes = NUM_CLASSES
		# model._build_yolo_head()
		# params, states = model.init_yolo_head(params, states, key)
  
		layer: Module
		model.set_trainable(True)
		for lidx, layer in enumerate(model.backbone.submodules.values()):
			layer.set_trainable(False)
			if lidx == 6: break

		loss_lambdas = {'lambda_box': 0.5, 'lambda_obj': 5, 'lambda_noobj': 1.5 , 'lambda_class': 1.5}
		hyperparameters = {
			'lr0' : 5e-4,
			'lr1' : 5e-5,
			'weight_decay' : 0.95
		}
   
		LR0 = hyperparameters['lr0']
		LR1 = hyperparameters['lr1']
		LR_DECAY = hyperparameters['weight_decay']

		# Initialize and train the model with the new lambda values
		params, states, yolo_loss, optimizer = initialize_training(
			params, states,
			model=model,
			learning_rate=hyperparameters['lr0'],
			loss_lambda=loss_lambdas
		)

		model, params, states, losses = train_model(
			params, states, optimizer, num_epochs=200, save_best_path=f'YOLOv3_RoadSign{iteration}.pkl'
		)