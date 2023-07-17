import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .bayes_cnn import BConvModule


class Multi_BNN(nn.Module):
	def __init__(self, out_channels=3, kernel_size=[1,3,5,7], nc=18, mixing=True, act='lrelu',
		clamp=True, num_blocks=4, a=0.5, data_mean=(0.5, 0.5, 0.5),
        data_std=(0.5, 0.5, 0.5)):
		super(Multi_BNN, self).__init__()
		if act == 'relu':
			self.act = nn.ReLU() 
		elif act == 'lrelu':
			self.act = nn.LeakyReLU(negative_slope=0.2)
		elif act == 'tanh':
			self.act = nn.Tanh()
		elif act == 'sigmoid':
			self.act == nn.Sigmoid() 
		# parameters
		self.mixing = mixing
		self.clamp = clamp
		self.a = a
		if num_blocks == 0:
			nc = 3
		#conv blocks
		self.blocks = []
		for i in range(num_blocks):
			if i == 0:
				self.blocks.append(
					BConvModule(
					in_channels=3,
					out_channels=nc,
					kernel_size=kernel_size,
					rand_bias=True))
			else:
				self.blocks.append(
					BConvModule(
						in_channels=nc,
						out_channels=nc,
						kernel_size=kernel_size,
						rand_bias=True)
					)



		self.final = BConvModule(
			in_channels=nc,
			out_channels=out_channels,
			kernel_size=kernel_size,
			mixing=mixing,
			identity_prob=0.0,
			rand_bias=False,
			data_mean=data_mean,
			data_std=data_std,
			clamp_output=clamp,
        )
		# blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
		# self.blocks = []
		# for bb in range(num_blocks):
		# 	self.blocks.append(blocks[bb])
		
	def forward(self, x):
		kl_loss = 0
		for bb in range(len(self.blocks)):
			x, kl = self.blocks[bb](x)
			x = self.act(x)
			kl_loss += kl
			if torch.isnan(x).any():
				print("layer {} problem".format(bb))
		out, kl = self.final(x)
		kl_loss += kl

		return out, kl_loss

	def randomize(self):
		for bb in range(len(self.blocks)):
			self.blocks[bb].randomize()
		self.final.randomize()


