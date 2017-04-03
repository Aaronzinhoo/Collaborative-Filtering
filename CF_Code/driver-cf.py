import random
import numpy as np
import scipy.sparse as sp
import cf

k = 40  # lower rank
Lambda = 0.1  # regularization parameter (the same for U and V)
num_iter = 20  # number of iterations in ALS

# Change 'personal_items' and 'personal_ratings' to your own moving ratings
# TODO
personal_items = np.array([81, 21, 180, 10, 567, 1038, 1036, 246, 351, 423])
personal_ratings = np.array([5, 4, 4, 5, 4, 2, 1, 2, 1, 1], dtype=np.float64)

# Read movie names and print your own ratings
num_users = 943
num_items = 1682
item_names = []
with open('ml-100k/u.item') as f:
	for line in f:
		item_names.append(line.split('|')[1])
assert len(personal_items) == len(personal_ratings)
print '==== Your personal ratings ===='
for i in xrange(len(personal_items)):
	print personal_ratings[i], item_names[personal_items[i]]
print ''

# Read train and test data, and create R_train and R_test in CSR sparse format
def read_file(filename):
	raw_data = np.genfromtxt(filename, dtype=np.int32)
	users = raw_data[:, 0] - 1
	items = raw_data[:, 1] - 1
	ratings = raw_data[:, 2].astype(np.float64)
	R = sp.coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
	return R.tocsr()
R_train = read_file('ml-100k/u2.base')
R_test = read_file('ml-100k/u2.test')

# Print baseline RMSE on test data by filling in global average
global_avg = R_train[R_train != 0].mean()
tmpU = np.ones((num_users, 1))
tmpV = global_avg * np.ones((1, num_items))
print 'RMSE with global average:', cf.compute_rmse(R_test, tmpU.dot(tmpV))

# Initialize U (k-by-m) and V (k-by-n)
U = np.zeros((k, num_users), order='F')
V = np.zeros((k, num_items), order='F')
U[:] = np.random.randn(k, num_users)
V[:] = np.random.randn(k, num_items)

# Call functions in cf for running ALS
print '==== Start running ALS ==== ({0:d} iterations, Lambda={1:g})'.format(num_iter, Lambda)
cf.ALS_train(R_train, U, V, Lambda, num_iter)
print 'RMSE on test data:', cf.compute_rmse_UV(R_test, U, V)
print ''

# Call functions in cf to make predictions based on your own ratings
# (the items in 'personal_items' is excluded from the recommendations)
personal_prediction = cf.ALS_predict(U, V, Lambda, personal_items, personal_ratings)
sorted_prediction = np.sort(personal_prediction)[::-1]
ind = np.argsort(personal_prediction)[::-1]
print 'Recommendations by ALS:'
count = 0
i = 0
while count < 20:
	if ind[i] not in personal_items:
		print personal_prediction[ind[i]], item_names[ind[i]]
		count += 1
	i += 1
