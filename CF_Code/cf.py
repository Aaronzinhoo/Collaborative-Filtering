import random
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# in the comments, m refers to number of users, n refers to number of items

def compute_rmse(R, prediction):
	# Compute RMSE based on a matrix filled with predicted ratings
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# prediction: Predicted ratings for all pairs of users and items, a dense matrix of size m-by-n
	# Return value -- RMSE
	# TODO
	# global estimation vs actual value reveal 1.131 (within range intended	x)
	pred = prediction.flatten()[:20000]
	R_sq = R.data*R.data
	pred_sq = pred*pred
	RP = -2*R.data*pred
	return  np.sqrt(np.mean(R_sq+pred_sq+RP))

def compute_rmse_UV(R, U, V):
	# Compute RMSE based on U and V
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Return value -- RMSE
	# TODO
	# creates vector with the corresponding values that arent zero in R
	R_sq = R.data*R.data 
	pred = U.T.dot(V)[R.toarray()!=0]
	# sqaures the terms for the RMSE value
	pred_sq = pred*pred
	RP = -2*pred*R.data
	return np.sqrt(np.mean(R_sq+pred_sq+RP))

def compute_cost_func(R, U, V, Lambda):
	# Compute the cost function in ALS (see details in homework instructions)
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Return value -- Value of the cost function
	# TODO

	# question to ask is that do we need to compute only the frob norm of values that dont correspond to zero in U and V
	r = R.toarray()
	U_V = U.T.dot(V)
	R_arr = r[r!=0]
	u_v = U_V[r!=0]
	
	# collect the frob portions of eq
	Frob_U = Lambda * np.linalg.norm(U,ord='fro')**2
	Frob_V = Lambda * np.linalg.norm(V,ord='fro')**2
	Frob  = np.sum((R_arr - u_v)**2)
	return Frob + Frob_V + Frob_U

def ALS_train(R, U, V, Lambda, num_iter):
	# Optimize the cost function and rewrite U and V
	# R: Incomplete rating matrix (m-by-n), a sparse matrix in CSR format ('0' means missing entries)
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Lambda: value of lambda in the cost function
	# num_iter: Number of outer iterations to run
	# (No return value)
	# TODO
	print R.nonzero()
	k = U.shape[0]
	m,n = R.shape
	L = Lambda*np.eye(k,dtype = np.float64)
		# to get the sparse matrices from before....
		# small letters used to denote vectors (except u)
	for _ in xrange(num_iter):
		for i in xrange(n):
			# get the pos in the first col where the values are nnz
			# R[:,i] gets tuples of the first col (row,col)
			# nonzero seperates the row/col index in tuple row->[0] ,col-> [1]
			# we ony want the row tuple so get [0] 
			nnz = R[:,i].nonzero()[0]
			# create R tilde fromt the notes now
			r = R[nnz,i].toarray()
			# create U tilde by taking cols that were observed entries in R (k x nnz) matrix 
			u = U[:,nnz]
			#print u.shape
			# get small v by solving for problem from slides (k) vector
			v = (np.linalg.inv(u.dot(u.T) + L).dot(u.dot(r))).reshape(k)
			V[:,i] = v
		for j in xrange(m):
			# get the pos in the first col where the values are nnz
			nnz = R[j,:].nonzero()[1]
			# create R tilde fromt the notes now
			r = R[j,nnz].toarray()
			# create U tilde by taking cols that were observed entries in R (k x nnz) matrix 
			v = V[:,nnz]
			#print v.shape
			# get small v by solving for problem from slides (k) vector
			u = (np.linalg.inv(v.dot(v.T) + L).dot(v.dot(r.T))).reshape(k)
			U[:,j] = u
		print compute_cost_func(R,U,V,Lambda)

def ALS_predict(U, V, Lambda, personal_items, personal_ratings):
	# Based on the model you've trained,
	# predict 	the ratings for all the items given your own ratings
	# U: k-by-m dense matrix
	# V: k-by-n dense matrix
	# Lambda: value of lambda in the cost function
	# personal_items: Indices of the items in your own ratings
	# personal_ratings: Your own ratings, corresponding to the indices in 'personal_items'
	# Return value: A n-vector containing the predicted ratings
	# TODO
	k, num_items = V.shape
	#personal_items = int(personal_items)
	new_user = np.zeros((num_items),dtype=np.float64).reshape(1,num_items)
	for i in xrange(10):
		new_user[0][int(personal_items[i])] = personal_ratings[i]
	nnz = new_user.nonzero()[1]
	# v is a k x ? matrix
	v = V[:,nnz]
	# VL_inv k x k matrix
	VL_inv = np.linalg.inv((v.dot(v.T))+Lambda*np.eye(k,dtype=np.float64))
	#add the new user to the estimated value of the original matrix 
	#vector that contains vals for Ui that is k x 1
	u = VL_inv.dot(v.dot(new_user[:,nnz].T))
	return u.T.dot(V).reshape(num_items)
