"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (N, M+1)
    """
    # TODO: Implement this function
    Phi = np.hstack([X**i for i in range(M+1)])
    return np.array(Phi)

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)
    
    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.
    """
    # TODO: Implement this function
    N=X.shape[0]  
    tests=X.dot(theta)  
    squared_loss=(y-tests)**2  
    loss=np.sum((squared_loss)/2)/(N)
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    N = len(y)
    tests = X.dot(theta)
    squared_errs = (y-tests)**2
    E_rms = np.sqrt(np.mean(squared_errs))  # Divide by N, then take the square root
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    theta = np.zeros(d)  
    prev_loss = float('inf')
    k = 0
    for _ in range(int(1e6)):  
        tests = X.dot(theta)
        err = tests-y
        gradient = X.T.dot(err)/N  
        theta -= learning_rate*gradient  
        new_loss = calculate_squared_loss(X, y, theta)
        if abs(new_loss-prev_loss) <= 1e-10:
            break
        
        prev_loss = new_loss
        k = k+1

    return theta, k+1

def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    theta = np.zeros(d)  
    prev_loss = float('inf')
    k = 0
    for _ in range(int(1e6)):
        if (learning_rate == 'adaptive'):
            lr = 1/(1+k)
        else:
            lr = learning_rate

        for i in range(N):
            tests = X[i].dot(theta)
            err = tests-y[i]
            gradient = X[i]*err
            
            theta -= lr*gradient 
            
        new_loss = calculate_squared_loss(X, y, theta)
        if abs(new_loss-prev_loss) <= 1e-10:
            break
        
        prev_loss = new_loss
        k = k+1


    return theta, k+1


def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    I = np.eye(d)
    theta = np.linalg.pinv(X.T.dot(X)+reg_param*I).dot(X.T).dot(y)

    return theta


def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    W = np.diag(weights)
    X_W_X = X.T @ W @ X
    X_W_y = X.T @ W @ y
    lambda_identity = reg_param*np.eye(X.shape[1])
    theta = np.linalg.inv(X_W_X + lambda_identity) @ X_W_y
    return theta


def part_1(fname_train):
    """
    This function should contain all the code you implement to complete part 1
    """
    print("========== Part 1 ==========")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    # Example of how to use the functions
    # start = time.process_time()
    # theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=0.01)
    # print('Time elapsed:', time.process_time() - start)

    # TODO: Add more code here to complete part 1
    ##############################
    X = Phi_train
    y = y_train
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    print(f"{'Algorithm':<20} {'η':<10} {'θ0':<10} {'θ1':<10} {'iterations':<15} {'Runtime (s)':<10}")
    for lr in learning_rates:
        start = time.time()
        theta_gd, k_gd = ls_gradient_descent(X, y, learning_rate=lr)
        runtime = time.time()-start
        iterations = k_gd 
        algo = "GD"
        print(f"{algo:<20} {lr:<10} {round(theta_gd[0], 5):<10} {round(theta_gd[1], 5):<10} {iterations:<15} {runtime:<10.4f}")

    for lr in learning_rates:
        start = time.time()
        theta_sgd, k_sgd = ls_stochastic_gradient_descent(X, y, learning_rate=lr)
        runtime = time.time()-start
        iterations = k_sgd  
        algo = "SGD"
        print(f"{algo:<20} {lr:<10} {round(theta_sgd[0], 5):<10} {round(theta_sgd[1], 5):<10} {iterations:<15} {runtime:<10.4f}")

    start = time.time()
    theta_cf = ls_closed_form_solution(X, y)
    runtime = time.time() - start
    algo = "Closed Form"
    print(f"{algo:<20} {0:<10} {round(theta_cf[0], 5):<10} {round(theta_cf[1], 5):<10} {0:<15} {runtime:<10.4f}")

    print("========== adaptive k ==========")
    start = time.time()
    theta_sgd, k_sgd = ls_stochastic_gradient_descent(X, y, learning_rate='adaptive')
    runtime = time.time()-start
    iterations = k_sgd  
    algo = "SGD"
    print(f"{algo:<20} {lr:<10} {round(theta_sgd[0], 5):<10} {round(theta_sgd[1], 5):<10} {iterations:<15} {runtime:<10.4f}")

    print("Done!")


def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """
    print("=========== Part 2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # TODO: Add more code here to complete part 2
    ##############################
    M_vals = np.arange(11)
    train_errs = []
    val_errs = []

    for M in M_vals:
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_validation = generate_polynomial_features(X_validation, M)
        theta = ls_closed_form_solution(Phi_train, y_train)
        train_rms_err = calculate_RMS_Error(Phi_train, y_train, theta)
        val_rms_err = calculate_RMS_Error(Phi_validation, y_validation, theta)
        train_errs.append(train_rms_err)
        val_errs.append(val_rms_err)

    plt.figure()
    plt.plot(M_vals, train_errs, label="Training Error", marker='o', color='red')
    plt.plot(M_vals, val_errs, label="Validation Error", marker='o', color='blue')
    plt.xlabel("Polynomial Degree M")
    plt.ylabel("RMS Error")
    plt.title("RMS Error vs. Polynomial Degree M")
    plt.legend()
    plt.grid(True)
    plt.show()

    #3.2.e
    M = 10
    Phi_train = generate_polynomial_features(X_train, M)
    Phi_val = generate_polynomial_features(X_validation, M)
    lambdas = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] #reg_param
    train_rms_errs = []
    val_rms_errs = []
    i = 0
    for reg_param in lambdas:
        theta = ls_closed_form_solution(Phi_train, y_train, reg_param=reg_param) 
        train_rms = calculate_RMS_Error(Phi_train, y_train, theta)
        train_rms_errs.append(train_rms)
        val_rms = calculate_RMS_Error(Phi_val, y_validation, theta)
        val_rms_errs.append(val_rms)

    plt.figure(figsize=(8, 6))
    plt.plot(np.log10(lambdas), train_rms_errs, label='Training Error', marker='o', color='red')
    plt.plot(np.log10(lambdas), val_rms_errs, label='Validation Error', marker='o', color='blue')
    plt.xlabel('Reg Param λ')
    plt.ylabel('RMS Error')
    plt.title('RMS Error vs. Regularization Parameter λ')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Done!")


def part_3(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 3
    """
    print("=========== Part 3 ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # TODO: Add more code here to complete part 3
    ##############################
    M_vals = np.arange(11)
    train_errs = []
    val_errs = []

    for M in M_vals:
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_validation = generate_polynomial_features(X_validation, M)
        theta = weighted_ls_closed_form_solution(Phi_train, y_train, weights_train)
        train_rms_err = calculate_RMS_Error(Phi_train, y_train, theta)
        val_rms_err = calculate_RMS_Error(Phi_validation, y_validation, theta)
        train_errs.append(train_rms_err)
        val_errs.append(val_rms_err)

    plt.figure()
    plt.plot(M_vals, train_errs, label="Training Error", marker='o', color='orange')
    plt.plot(M_vals, val_errs, label="Validation Error", marker='o', color='green')
    plt.xlabel("Polynomial Degree (M)")
    plt.ylabel("RMS Error")
    plt.title("RMS Error vs. Polynomial Degree M (with weights)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Done!")


def main(fname_train, fname_validation):
    part_1(fname_train)
    part_2(fname_train, fname_validation)
    part_3(fname_train, fname_validation)


if __name__ == '__main__':
    main("dataset/linreg_train.csv", "dataset/linreg_validation.csv")
