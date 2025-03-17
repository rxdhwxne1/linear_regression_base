from numpy import *
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        totalError += (y - (m * x + b)) ** 2
        
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    
    # to stock the evolution of the regression
    history = {'b': [b], 'm': [m], 'error': [compute_error_for_line_given_points(b, m, points)]}
    
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        
        # register the evolution of the regression every 500 iterations
        if i % 500 == 0:
            error = compute_error_for_line_given_points(b, m, points)
            history['b'].append(b)
            history['m'].append(m)
            history['error'].append(error)
            
    return [b, m, history]

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def plot_regression_evolution(points, history):
    plt.figure(figsize=(12, 10))
    
    # Subplot for the data and the regression line
    plt.subplot(2, 1, 1)
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Data')
    
    # draw multiple regression lines
    x = linspace(min(points[:, 0]), max(points[:, 0]), 500)
    colors = plt.cm.jet(linspace(0, 1, len(history['b'])))
    
    for i, (b, m) in enumerate(zip(history['b'], history['m'])):
        y = m * x + b
        plt.plot(x, y, color=colors[i], alpha=0.7, 
                 label=f'Iteration {i*500}' if i > 0 else 'Initial')
    
    plt.title('Linear Regression Evolution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Subplot for the error evolution
    plt.subplot(2, 1, 2)
    plt.plot(range(0, len(history['error'])*500, 500), history['error'], 
             marker='o', linestyle='-', color='red')
    plt.title('Error Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run():
    # step 1 
    points = genfromtxt("data.csv", delimiter=",")
    
    # step 2 - define hyperparameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    
    # step 3 - train our model
    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m, history] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    print('ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    # step 4 - visualize the evolution
    plot_regression_evolution(points, history)

if __name__ == '__main__':
    run()