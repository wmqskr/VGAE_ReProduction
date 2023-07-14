import mindspore
import mindspore.nn as nn
import os
import numpy as np
import args

class VGAE(nn.Cell):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		#编码
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = mindspore.ops.randn(X.size(0), args.hidden2_dim)
		sampled_z = gaussian_noise*mindspore.ops.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class GraphConvSparse(nn.Cell):
	#定义图卷积层
	def __init__(self, input_dim, output_dim, adj, activation = mindspore.ops.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)	#图卷积层的权重
		self.adj = adj	#图的邻接矩阵
		self.activation = activation	#激活函数

	def forward(self, inputs):
		#图卷积操作的前向传播
		x = inputs
		x = mindspore.ops.mm(x, self.weight)
		x = mindspore.ops.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	#图自编码器模型的解码操作
	A_pred = mindspore.ops.sigmoid(mindspore.ops.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	#参数初始化
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = mindspore.ops.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Cell):
	def __init__(self,adj):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred
		