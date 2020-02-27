import os
import time
import math
import random
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import h5py
import shutil
import mcubes
from bspt import digest_bsp, get_mesh, get_mesh_watertight
#from bspt_slow import digest_bsp, get_mesh, get_mesh_watertight

from utils import *

#tensorflow 2.0 implementation


class generator(Model):
	def __init__(self, phase, p_dim, c_dim):
		super(generator, self).__init__()
		self.phase = phase
		self.p_dim = p_dim
		self.c_dim = c_dim
		initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
		initializer2 = tf.keras.initializers.RandomNormal(mean=1e-5, stddev=0.02)
		if self.phase==0:
			self.convex_layer_weights = tf.Variable(initializer(shape=[self.p_dim, self.c_dim]), name="convex_layer_weights")
			self.concave_layer_weights = tf.Variable(initializer2(shape=[self.c_dim, 1]), name="concave_layer_weights")
		elif self.phase==1 or self.phase==2:
			self.convex_layer_weights = tf.Variable(initializer(shape=[self.p_dim, self.c_dim]), name="convex_layer_weights", trainable=False)
			self.concave_layer_weights = tf.Variable(initializer2(shape=[self.c_dim, 1]), name="concave_layer_weights", trainable=False)
		elif self.phase==3:
			self.convex_layer_weights = tf.Variable(initializer(shape=[self.p_dim, self.c_dim]), name="convex_layer_weights", trainable=True)
			self.concave_layer_weights = tf.Variable(initializer2(shape=[self.c_dim, 1]), name="concave_layer_weights", trainable=False)

	def call(self, points, plane_m, is_training=False):
		if self.phase==0:
			#level 1
			h1 = tf.linalg.matmul(points, plane_m)
			h1 = tf.math.maximum(h1, 0)

			#level 2
			h2 = tf.linalg.matmul(h1, self.convex_layer_weights)
			h2 = tf.math.maximum(tf.math.minimum(1-h2, 1), 0)

			#level 3
			h3 = tf.linalg.matmul(h2, self.concave_layer_weights)
			h3 = tf.math.maximum(tf.math.minimum(h3, 1), 0)

			return h2,h3
		elif self.phase==1 or self.phase==2:
			#level 1
			h1 = tf.linalg.matmul(points, plane_m)
			h1 = tf.math.maximum(h1, 0)

			#level 2
			h2 = tf.linalg.matmul(h1, tf.cast(self.convex_layer_weights>0.01, self.convex_layer_weights.dtype))

			#level 3
			h3 = tf.math.reduce_min(h2, axis=2, keepdims=True)

			return h2,h3
		elif self.phase==3:
			#level 1
			h1 = tf.linalg.matmul(points, plane_m)
			h1 = tf.math.maximum(h1, 0)

			#level 2
			h2 = tf.linalg.matmul(h1, self.convex_layer_weights)

			#level 3
			h3 = tf.math.reduce_min(h2, axis=2, keepdims=True)

			return h2,h3
		else:
			print("Congrats you got an error!")
			print("generator.phase should be in [0,1,2,3], got", self.phase)
			exit(0)

class encoder(Model):
	def __init__(self, ef_dim):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		initializer = tf.keras.initializers.GlorotUniform()
		self.conv_1 = layers.Conv3D(self.ef_dim, 4, strides=2, padding="same", use_bias=True, kernel_initializer=initializer)
		self.conv_2 = layers.Conv3D(self.ef_dim*2, 4, strides=2, padding="same", use_bias=True, kernel_initializer=initializer)
		self.conv_3 = layers.Conv3D(self.ef_dim*4, 4, strides=2, padding="same", use_bias=True, kernel_initializer=initializer)
		self.conv_4 = layers.Conv3D(self.ef_dim*8, 4, strides=2, padding="same", use_bias=True, kernel_initializer=initializer)
		self.conv_5 = layers.Conv3D(self.ef_dim*8, 4, strides=1, padding="valid", use_bias=True, kernel_initializer=initializer)

	def call(self, inputs, is_training=False):
		d_1 = self.conv_1(inputs)
		d_1 = tf.nn.leaky_relu(d_1, alpha=0.01)

		d_2 = self.conv_2(d_1)
		d_2 = tf.nn.leaky_relu(d_2, alpha=0.01)
		
		d_3 = self.conv_3(d_2)
		d_3 = tf.nn.leaky_relu(d_3, alpha=0.01)

		d_4 = self.conv_4(d_3)
		d_4 = tf.nn.leaky_relu(d_4, alpha=0.01)

		d_5 = self.conv_5(d_4)
		d_5 = tf.reshape(d_5,[-1, self.ef_dim*8])
		d_5 = tf.math.sigmoid(d_5)

		return d_5

class decoder(Model):
	def __init__(self, ef_dim, p_dim):
		super(decoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		initializer = tf.keras.initializers.GlorotUniform()
		self.linear_1 = layers.Dense(self.ef_dim*16, use_bias=True, kernel_initializer=initializer)
		self.linear_2 = layers.Dense(self.ef_dim*32, use_bias=True, kernel_initializer=initializer)
		self.linear_3 = layers.Dense(self.ef_dim*64, use_bias=True, kernel_initializer=initializer)
		self.linear_4 = layers.Dense(self.p_dim*4, use_bias=True,  kernel_initializer=initializer)

	def call(self, inputs, is_training=False):
		l1 = self.linear_1(inputs)
		l1 = tf.nn.leaky_relu(l1, alpha=0.01)

		l2 = self.linear_2(l1)
		l2 = tf.nn.leaky_relu(l2, alpha=0.01)

		l3 = self.linear_3(l2)
		l3 = tf.nn.leaky_relu(l3, alpha=0.01)

		l4 = self.linear_4(l3)
		l4 = tf.reshape(l4, [-1, 4, self.p_dim])

		return l4

class bsp_network(Model):
	def __init__(self, phase, ef_dim, p_dim, c_dim):
		super(bsp_network, self).__init__()
		self.phase = phase
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.c_dim = c_dim
		self.encoder = encoder(self.ef_dim)
		self.decoder = decoder(self.ef_dim, self.p_dim)
		self.generator = generator(self.phase, self.p_dim, self.c_dim)

	def call(self, inputs, z_vector, plane_m, point_coord, is_training=False):
		if is_training:
			z_vector = self.encoder(inputs, is_training=is_training)
			plane_m = self.decoder(z_vector, is_training=is_training)
			net_out_convexes, net_out = self.generator(point_coord, plane_m, is_training=is_training)
		else:
			if inputs is not None:
				z_vector = self.encoder(inputs, is_training=is_training)
			if z_vector is not None:
				plane_m = self.decoder(z_vector, is_training=is_training)
			if point_coord is not None:
				net_out_convexes, net_out = self.generator(point_coord, plane_m, is_training=is_training)
			else:
				net_out_convexes = None
				net_out = None

		return z_vector, plane_m, net_out_convexes, net_out


class BSP_AE(object):
	def __init__(self, config):
		"""
		Args:
			too lazy to explain
		"""
		self.phase = config.phase

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.sample_vox_size = config.sample_vox_size
		if self.sample_vox_size==16:
			self.load_point_batch_size = 16*16*16
		elif self.sample_vox_size==32:
			self.load_point_batch_size = 16*16*16
		elif self.sample_vox_size==64:
			self.load_point_batch_size = 16*16*16*4
		self.shape_batch_size = 18
		self.point_batch_size = 16*16*16
		self.input_size = 64 #input voxel grid size

		self.ef_dim = 32
		self.p_dim = 4096
		self.c_dim = 256

		self.dataset_name = config.dataset
		self.dataset_load = self.dataset_name + '_train'
		if not (config.train or config.getz):
			self.dataset_load = self.dataset_name + '_test'
		self.checkpoint_dir = config.checkpoint_dir
		self.data_dir = config.data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
		if os.path.exists(data_hdf5_name):
			data_dict = h5py.File(data_hdf5_name, 'r')
			self.data_points = ((data_dict['points_'+str(self.sample_vox_size)][:]).astype(np.float32)+0.5)/256-0.5
			self.data_points = np.concatenate([self.data_points, np.ones([len(self.data_points),self.load_point_batch_size,1],np.float32) ],axis=2)
			self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
			self.data_voxels = data_dict['voxels'][:].astype(np.float32)
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		
		self.real_size = 64 #output point-value voxel grid size in testing
		self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
		test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change
		
		#get coords
		dima = self.test_size
		dim = self.real_size
		self.aux_x = np.zeros([dima,dima,dima],np.uint8)
		self.aux_y = np.zeros([dima,dima,dima],np.uint8)
		self.aux_z = np.zeros([dima,dima,dima],np.uint8)
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					self.aux_x[i,j,k] = i*multiplier
					self.aux_y[i,j,k] = j*multiplier
					self.aux_z[i,j,k] = k*multiplier
		self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
		self.coords = (self.coords+0.5)/dim-0.5
		self.coords = np.reshape(self.coords,[multiplier3,test_point_batch_size,3])
		self.coords = np.concatenate([self.coords, np.ones([multiplier3,test_point_batch_size,1],np.float32) ],axis=2)

		#build model
		self.bsp_network = bsp_network(config.phase, self.ef_dim, self.p_dim, self.c_dim)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1)
		self.checkpoint = tf.train.Checkpoint(encoder=self.bsp_network.encoder,decoder=self.bsp_network.decoder,generator=self.bsp_network.generator)
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=2, checkpoint_name='BSP_AE.model')
		#loss
		if config.phase==0:
			#phase 0 continuous for better convergence
			#L_recon + L_W + L_T
			#G2 - network output (convex layer), the last dim is the number of convexes
			#G - network output (final output)
			#point_value - ground truth inside-outside value for each point
			#cw2 - connections T
			#cw3 - auxiliary weights W
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = tf.math.reduce_mean(tf.math.square(point_value - G))
				loss = loss_sp + tf.math.reduce_sum(tf.math.abs(cw3-1)) + (tf.math.reduce_sum(tf.math.maximum(cw2-1,0)) - tf.math.reduce_sum(tf.math.minimum(cw2,0)))
				return loss_sp,loss
			self.loss = network_loss
		elif config.phase==1:
			#phase 1 hard discrete for bsp
			#L_recon
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = tf.math.reduce_mean((1-point_value)*(1-tf.math.minimum(G,1)) + point_value*(tf.math.maximum(G,0)))
				loss = loss_sp
				return loss_sp,loss
			self.loss = network_loss
		elif config.phase==2:
			#phase 2 hard discrete for bsp with L_overlap
			#L_recon + L_overlap
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = tf.math.reduce_mean((1-point_value)*(1-tf.math.minimum(G,1)) + point_value*(tf.math.maximum(G,0)))
				G2_inside = tf.cast(G2<0.01, tf.float32)
				bmask = G2_inside * tf.cast(tf.reduce_sum(G2_inside, axis=2, keepdims=True)>1, tf.float32)
				loss = loss_sp - tf.math.reduce_mean(G2*point_value*bmask)
				return loss_sp,loss
			self.loss = network_loss
		elif config.phase==3:
			#phase 3 soft discrete for bsp
			#L_recon + L_T
			#soft cut with loss L_T: gradually move the values in T (cw2) to either 0 or 1
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = tf.math.reduce_mean((1-point_value)*(1-tf.math.minimum(G,1)) + point_value*(tf.math.maximum(G,0)))
				loss = loss_sp + (tf.math.reduce_sum(tf.math.minimum(tf.math.abs(cw2)*100,tf.math.abs(cw2-1))))
				return loss_sp,loss
			self.loss = network_loss

	@property
	def model_dir(self):
		return "{}_ae_{}".format(self.dataset_name, self.input_size)

	def train(self, config):
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			#manually backup checkpoints because tensorflow CheckpointManager will delete all your previous checkpoints. Who designed this CheckpointManager??
			counter = 0
			new_folder_name = "backup-before-training-ae-"+str(self.sample_vox_size)+"-"+str(counter)
			while os.path.exists(os.path.join(self.checkpoint_dir, new_folder_name)):
				counter += 1
				new_folder_name = "backup-before-training-ae-"+str(self.sample_vox_size)+"-"+str(counter)
			shutil.copytree(self.checkpoint_path, os.path.join(self.checkpoint_dir, new_folder_name))
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			
		shape_num = len(self.data_voxels)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)
		point_batch_num = int(self.load_point_batch_size/self.point_batch_size)

		for epoch in range(0, training_epoch):
			np.random.shuffle(batch_index_list)
			avg_loss_sp = 0
			avg_loss_tt = 0
			avg_num = 0
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
				batch_voxels = self.data_voxels[dxb]
				if point_batch_num==1:
					point_coord = self.data_points[dxb]
					point_value = self.data_values[dxb]
				else:
					which_batch = np.random.randint(point_batch_num)
					point_coord = self.data_points[dxb,which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
					point_value = self.data_values[dxb,which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]

				with tf.GradientTape() as ggg:
					_,_, net_out_convexes, net_out = self.bsp_network(batch_voxels, None, None, point_coord, is_training=True)
					errSP, errTT = self.loss(net_out_convexes, net_out, point_value, self.bsp_network.generator.convex_layer_weights, self.bsp_network.generator.concave_layer_weights)

				net_gradients = ggg.gradient(errTT, self.bsp_network.trainable_variables)
				self.optimizer.apply_gradients(zip(net_gradients, self.bsp_network.trainable_variables))

				avg_loss_sp += errSP
				avg_loss_tt += errTT
				avg_num += 1
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f, loss_total: %.6f" % (epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num, avg_loss_tt/avg_num))
			if epoch%10==9:
				self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			if epoch%20==19:
				self.checkpoint_manager.save()

		self.checkpoint_manager.save()

	def test_1(self, config, name):
		multiplier = int(self.real_size/self.test_size)
		multiplier2 = multiplier*multiplier

		if config.phase==0:
			thres = 0.5
		else:
			thres = 0.99
		
		t = np.random.randint(len(self.data_voxels))
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		batch_voxels = self.data_voxels[t:t+1]
		_, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					minib = i*multiplier2+j*multiplier+k
					point_coord = self.coords[minib:minib+1]
					_,_,_, net_out = self.bsp_network(None, None, out_m, point_coord, is_training=False)
					if config.phase!=0:
						net_out = tf.math.maximum(tf.math.minimum(1-net_out, 1), 0)
					model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(net_out, [self.test_size,self.test_size,self.test_size])
		
		vertices, triangles = mcubes.marching_cubes(model_float, thres)
		vertices = (vertices-0.5)/self.real_size-0.5
		#output ply sum
		write_ply_triangle(config.sample_dir+"/"+name+".ply", vertices, triangles)
		print("[sample]")


	#output bsp shape as ply
	def test_bsp(self, config):
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		w2 = self.bsp_network.generator.convex_layer_weights.numpy()

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		for t in range(config.start, min(len(self.data_voxels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			_, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.c_dim])
			
			out_m = out_m.numpy()

			bsp_convex_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						model_float_sum = model_float_sum-slice_i
					else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_m[0,3,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			vertices, polygons = get_mesh(bsp_convex_list)
			#use the following alternative to merge nearby vertices to get watertight meshes
			#vertices, polygons = get_mesh_watertight(bsp_convex_list)

			#output ply
			write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
	
	#output bsp shape as ply and point cloud as ply
	def test_mesh_point(self, config):
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		w2 = self.bsp_network.generator.convex_layer_weights.numpy()
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		for t in range(config.start, min(len(self.data_voxels),config.end)):
			print(t)
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			model_float_combined = np.ones([self.real_size,self.real_size,self.real_size],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			_, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_, model_out, model_out_combined = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.c_dim])
						model_float_combined[self.aux_x+i,self.aux_y+j,self.aux_z+k] = np.reshape(model_out_combined, [self.test_size,self.test_size,self.test_size])
			
			out_m = out_m.numpy()

			bsp_convex_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					#if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
					#	model_float_sum = model_float_sum-slice_i
					#else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_m[0,3,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))
							
			#convert bspt to mesh
			vertices, polygons = get_mesh(bsp_convex_list)
			#use the following alternative to merge nearby vertices to get watertight meshes
			#vertices, polygons = get_mesh_watertight(bsp_convex_list)

			#output ply
			write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
			
			#sample surface points
			sampled_points_normals = sample_points_polygon_vox64(vertices, polygons, model_float_combined, 16000)
			#check point inside shape or not
			point_coord = np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3])
			point_coord = np.concatenate([point_coord, np.ones([1,point_coord.shape[1],1],np.float32) ],axis=2)
			_,_,_, sample_points_value = self.bsp_network(None, None, out_m, point_coord, is_training=False)
			sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
			print(len(bsp_convex_list), len(sampled_points_normals))
			np.random.shuffle(sampled_points_normals)
			write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals[:4096])


	#output bsp shape as obj with color
	def test_mesh_obj_material(self, config):
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		w2 = self.bsp_network.generator.convex_layer_weights.numpy()

		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier

		#write material
		#all output shapes share the same material
		#which means the same convex always has the same color for different shapes
		#change the colors in default.mtl to visualize correspondences between shapes
		fout2 = open(config.sample_dir+"/default.mtl", 'w')
		for i in range(self.c_dim):
			fout2.write("newmtl m"+str(i+1)+"\n") #material id
			fout2.write("Kd 0.80 0.80 0.80\n") #color (diffuse) RGB 0.00-1.00
			fout2.write("Ka 0 0 0\n") #color (ambient) leave 0s
		fout2.close()


		for t in range(config.start, min(len(self.data_voxels),config.end)):
			model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			_, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.c_dim])
			
			out_m = out_m.numpy()
			
			bsp_convex_list = []
			color_idx_list = []
			model_float = model_float<0.01
			model_float_sum = np.sum(model_float,axis=3)
			for i in range(self.c_dim):
				slice_i = model_float[:,:,:,i]
				if np.max(slice_i)>0: #if one voxel is inside a convex
					if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
						model_float_sum = model_float_sum-slice_i
					else:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[0,0,j]
								b = -out_m[0,1,j]
								c = -out_m[0,2,j]
								d = -out_m[0,3,j]
								box.append([a,b,c,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))
							color_idx_list.append(i)

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			vertices = []

			#write obj
			fout2 = open(config.sample_dir+"/"+str(t)+"_bsp.obj", 'w')
			fout2.write("mtllib default.mtl\n")

			for i in range(len(bsp_convex_list)):
				vg, tg = get_mesh([bsp_convex_list[i]])
				vbias=len(vertices)+1
				vertices = vertices+vg

				fout2.write("usemtl m"+str(color_idx_list[i]+1)+"\n")
				for ii in range(len(vg)):
					fout2.write("v "+str(vg[ii][0])+" "+str(vg[ii][1])+" "+str(vg[ii][2])+"\n")
				for ii in range(len(tg)):
					fout2.write("f")
					for jj in range(len(tg[ii])):
						fout2.write(" "+str(tg[ii][jj]+vbias))
					fout2.write("\n")

			fout2.close()


	#output h3
	def test_dae3(self, config):
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		
		for t in range(config.start, min(len(self.data_voxels),config.end)):
			model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
			batch_voxels = self.data_voxels[t:t+1]
			_, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_,_,_, model_out = self.bsp_network(None, None, out_m, point_coord, is_training=False)
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
			
			vertices, triangles = mcubes.marching_cubes(model_float, 0.5)
			vertices = (vertices-0.5)/self.real_size-0.5
			#output prediction
			write_ply_triangle(config.sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)

			vertices, triangles = mcubes.marching_cubes(batch_voxels[0,:,:,:,0], 0.5)
			vertices = (vertices-0.5)/self.real_size-0.5
			#output ground truth
			write_ply_triangle(config.sample_dir+"/"+str(t)+"_gt.ply", vertices, triangles)
			
			print("[sample]")
	
	def get_z(self, config):
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		hdf5_path = self.checkpoint_dir+'/'+self.model_dir+'/'+self.dataset_name+'_train_z.hdf5'
		shape_num = len(self.data_voxels)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		hdf5_file.create_dataset("zs", [shape_num,self.ef_dim*8], np.float32)

		print(shape_num)
		for idx in range(shape_num):
			batch_voxels = self.data_voxels[idx:idx+1]
			out_z, _,_,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
			hdf5_file["zs"][idx:idx+1,:] = out_z

		hdf5_file.close()
		print("[z]")

