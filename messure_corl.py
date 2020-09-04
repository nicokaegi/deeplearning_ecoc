import numpy as np
import sys
import math
import os

def count_bothAandB(i_lables, i_outputs, j_lables, j_outputs):

	AandB = 0

	pos = 0

	while(pos < i_lables.shape[0]):

		if( (i_lables[pos] == i_outputs[pos]) and (j_lables[pos] == j_outputs[pos]) ):
			AandB += 1

		pos += 1

	return AandB/i_lables.shape[0]

# todo : compute the avergae correlation, std, and seperatet means, ans stds for negative, and pos

def main():

	results_path = "/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/" + sys.argv[1] + "/results/{}{}"
	testdata_path = "/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/" + sys.argv[1] + "/test_{}/{}/test_lables.npy"

	test = sys.argv[2][0]
	variant =sys.argv[2][3]

	outputs = np.load(results_path.format(sys.argv[2],"_output_array.npy"))
	mean_time = np.load(results_path.format(sys.argv[2],"_average_times.npy"))
	total_time = np.load(results_path.format(sys.argv[2],"_total_times.npy"))
	accuracy_data = np.load(results_path.format(sys.argv[2],"_accuracy_array.npy"))

	test_lables = np.load("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/" + sys.argv[1] + "/test_1/{}/test_lables.npy".format(variant))

	target_matrix = np.ones( (len(accuracy_data), len(accuracy_data)))

	correlations = []
	pos_correlations = []
	neg_correlations = []

	i = 0
	while(i < target_matrix.shape[0]):

		q_i = 1 - accuracy_data[i]

		j = 0
		while(j < target_matrix.shape[1]):

			if( i != j):

				AandB = count_bothAandB( test_lables[:,i], outputs[:,i]  ,test_lables[:,j]  ,outputs[:,j])
				q_j = 1 - accuracy_data[j]
				itimesj = accuracy_data[i]*accuracy_data[j]
				target_matrix[i,j] = (AandB - itimesj)/math.sqrt(accuracy_data[i]*q_i*accuracy_data[j]*q_j)

				correlations.append(target_matrix[i,j])

				if(target_matrix[i,j] >= 0):
					pos_correlations.append(target_matrix[i,j])

				else:
					neg_correlations.append(target_matrix[i,j])

			j += 1

		i += 1

	correlations = np.array(correlations)
	pos_correlations = np.array(pos_correlations)
	neg_correlations = np.array(neg_correlations)

	print(correlations.mean(), correlations.std())

	np.savetxt("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/{}_mean_std.csv".format(sys.argv[2]), np.array([correlations.mean(), correlations.std()]), delimiter=',')

	print(pos_correlations.mean(), pos_correlations.std())
	np.savetxt("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/{}_pos_mean_std.csv".format(sys.argv[2]), np.array([pos_correlations.mean(), pos_correlations.std()]), delimiter=',')

	print(neg_correlations.mean(), neg_correlations.std())
	np.savetxt("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/{}_neg_mean_std.csv".format(sys.argv[2]), np.array([neg_correlations.mean(), neg_correlations.std()]), delimiter=',')


	np.savetxt("/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/data/{}_correlation_matrix.csv".format(sys.argv[2]),target_matrix, delimiter=',')

if __name__ == "__main__":
	main()
