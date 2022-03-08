from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)

    # ------------Kernel Logistic Regression Case------------
    ### YOUR CODE HERE
    # Run your kernel logistic regression model here
    learning_rate = [0.01,0.001]
	max_epoch = [30, 50]
    batch_size = [64, 128]
    sigma = [0.001,0.01,0.1,1,10,100,1000]
	scores=[]
    max_acc = 0.0
    hyper_params = []
   
    for learning in learning_rate:
         for epoch in max_epoch:
            for size_b in batch_size:
                for sigma_val in sigma:
                    model = Model('Kernel_LR', train_X.shape[0], sigma_val)
                    model.train(train_X, train_y, valid_X, valid_y, epoch, learning, size_b)
                    score = model.score(valid_X, valid_y)
                    hyper_params.append([score, learning, epoch, size_b, sigma_val]) 
                    scores.append([sigma_val,learning,score])
                    print("score = {} in test set.\n".format(score))
    print(scores)
	print("===========best hyper params===============")
    print(sorted(hyper_params, key=lambda x:x[0]))    
    
    model = Model('Kernel_LR', train_valid_X.shape[0], 10)
    model.train(train_valid_X, train_valid_y, None, None, 30, 0.01, 64)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE
​
​
    # ------------RBF Network Case------------
    ### YOUR CODE HERE
    # Run your radial basis function network model here
    
    learning_rate = [0.01,0.001]
    max_epoch = [30,50]
    batch_size = [64, 128]
    sigma = [0.001,0.01,0.1,1,10,100,1000]
    scores=[]
    hidden_dim=[12, 15, 18]
    hyper_params = []
    
    for learning in learning_rate:
        for epoch in max_epoch:
            for size_b in batch_size:
                for sigma_val in sigma:
                    for hidden in hidden_dim:
                        model = Model('RBF', hidden, sigma_val)
                        model.train(train_X, train_y, valid_X, valid_y, epoch, learning, size_b)
                        score = model.score(test_X, test_y)
                        hyper_params.append([score, learning, epoch, size_b, sigma_val, hidden]) 
                        scores.append([sigma_val,learning,score])
                        print("score = {} in test set.\n".format(score))
    print(scores)
    print("===========sorted hyper params===============")
    print(sorted(hyper_params, key=lambda x:x[0], reverse=True)) 
    
​	# test with best hyperparams
    model = Model('RBF', 12, 10)
    model.train(train_valid_X, train_valid_y, None, None, 30, 0.01, 64)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE
​
    # ------------Feed-Forward Network Case------------
    ### YOUR CODE HERE
    # Run your feed-forward network model here
   
    learning_rate = [0.01,0.001]
    max_epoch = [30, 50]
    batch_size = [64, 128]
    sigma = [0.001,0.01,0.1,1,10,100,1000]
    scores=[]
    hidden_dimension=[1,10,50,100]
    #output = torch.empty(train_X.shape[0],self.prototypes.shape[0])
    for learning in learning_rate:
        for epoch in max_epoch:
            for size_b in batch_size:
                for hidden_dim in hidden_dimension:
                    model = Model('FFN', hidden_dim)
                    model.train(train_X, train_y, valid_X, valid_y, epoch, learning, size_b)
                    score = model.score(test_X, test_y)
                    scores.append([learning,score])
                    print("score = {} in test set.\n".format(score))
    print(scores)
​
    # test with best hyperparams
    model = Model('FFN', 12)
    model.train(train_valid_X, train_valid_y, None, None, 30, 0.01, 64)
    score = model.score(test_X, test_y)
    print("score = {} in test set\n".format(score))
    ### END YOUR CODE
    
if __name__ == '__main__':
    main()