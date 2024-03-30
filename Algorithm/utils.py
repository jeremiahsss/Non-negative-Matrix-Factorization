import numpy as np
from noise import Noise
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from l2_norm_NMF import nmf_frobenius 
from l1_norm_NMF import l1_norm_nmf
from CIM_NMF import cim_nmf
from huber_nmf import huber_nmf
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from tabulate import tabulate



def noise_demo(X_hat, noise_list, noise_name):
    """
    Demonstrate demo of images before and after contaminated

    Parameters:
    - X_hat: Original image list
    - noise_list: Generated list
    - noise_name: A list of the detail of noise type and level 
    """
    X_hat_list = []
    X_noise_list = []
    for params in noise_list:
        # Unpack the dictionary and pass it as arguments
        X_hat_noise, X_noise = Noise(**params).generate_noise()
        # Store the contaminated iamge into the corresponding list
        X_hat_list.append(X_hat_noise)
        X_noise_list.append(X_noise)
    
    # Display noise
    img_size = [i//3 for i in (92, 112)] 
    ind = 2 # index of demo image.
    for i in range(len(noise_name)):
        plt.figure(figsize=(15,15))
        plt.subplot(731)
        plt.imshow(X_hat[:,ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
        plt.title('Image(Original)')
        plt.subplot(732)
        plt.imshow(X_noise_list[i][:,ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
        plt.title(f"Noise: {noise_name[i]}")
        plt.subplot(733)
        plt.imshow(X_hat_list[i][:,ind].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
        plt.title(f'Image({noise_name[i]})')
        plt.tight_layout()
        plt.show()


def display_reconstructed_images(X, Title, img_shape, n_images=5):
    """
    Display the original and reconstructed images side by side.

    Parameters:
    - D: Basis matrix
    - R: Coefficient matrix
    - img_shape: Shape of the images (e.g., (48, 42))
    - n_images: Number of images to display (default is 5)
    """
    plt.figure(figsize=(2*n_images, 4)) # Adjust the size to display properly
    
    for i in range(n_images):
        # Original Image
        plt.subplot(1, n_images, i + 1)
        plt.imshow(X[:, i].reshape(img_shape), cmap=plt.cm.gray)
        plt.title(Title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


    # Create noise
def generate_noise(img, img_shape, noise_type, **kwargs):
    """
    Generate noise upon given images

    Parameters:
    - img: The images to be contaminated
    - img_shape: The shape of one image
    - noise_type: Used to set up the type of noise
    """
    noise_generator = Noise(img=img, img_shape=img_shape, noise_type=noise_type, **kwargs)
    
    X, X_noise = noise_generator.generate_noise()
    
    # return the contaminated images
    return X


def split_data_set(dataset, train_ratio = 0.9, random_state = 42, **kwargs):
    """
    Split dataset into training and testing data.

    Parameters:
    - dataset: The dataset to be split
    - train_ratio: The proportion of training dataset
    - random_state: Used to initialize the random number generator
    """
    np.random.seed(random_state)
    
    train_ratio = train_ratio
    test_ratio = 1.0 - train_ratio
    #shuffle indices and split
    indices = np.random.permutation(len(dataset))
    split_index = int(train_ratio * len(dataset))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    #split the data base on the indices
    train_set = dataset[train_indices]
    test_set = dataset[test_indices]

    return train_set, test_set


def assign_cluster_label(X, Y):
    """
    Assign labels to sample based on  K-means clustering

    Parameters:
    - X: The attribute list of data
    - Y: The ture label of data used to initialization
    """
    # fit kmeans
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    # assign predicted labels based on kmeans
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred


# write a function to calculate the accuracy and nmi
def calculate_acc_nmi(R, Y_hat):
    """
    Calculate accuracy and NMI 

    Parameters:
    - R: The reconstructed matrix of data 
    - Y_hat: The ture label of data 
    """
    # Predict label of R
    Y_pred = assign_cluster_label(R.T, Y_hat)
    # calculate the result 
    acc = accuracy_score(Y_hat, Y_pred)
    nmi = normalized_mutual_info_score(Y_hat, Y_pred)
    return acc, nmi

def get_training_list(X_hat, Y_hat, cv = 3):
    """
    Randomly select 90% of the dataset and store them in a list

    Parameters:
    - X_hat: The attribute list of data
    - Y_hat: The label list of data
    - cv: The number of corss validtaion
    """
    X_hat_list = []
    Y_hat_list = []
    # Randomly select 90% of the dataset multiple times
    for i in range(cv):
        temp_X_hat, _ = split_data_set(X_hat.T, train_ratio = 0.9, random_state = i)
        temp_X_hat = temp_X_hat.T
        temp_Y_hat, _ = split_data_set(Y_hat, train_ratio = 0.9, random_state = i)

        X_hat_list.append(temp_X_hat)
        Y_hat_list.append(temp_Y_hat)

    return X_hat_list, Y_hat_list


def generate_test_image(X_hat_list, img_shape):

    """
    Create a list that include maps from noise type to generated noisy images

    Parameters:
    - X_hat_list: The list of validation dataset
    - img_shape: The shape of image
    """
    X_hat_noisy_maplist = []

    for X_hat in X_hat_list:
        # Ceate 3 block noise images
        X_block_04 = generate_noise(X_hat, img_shape, noise_type="block", b=4)
        X_block_08 = generate_noise(X_hat, img_shape, noise_type="block", b=8)
        X_block_12 = generate_noise(X_hat, img_shape, noise_type="block", b=12)

        # Create 3 saltpepper noise images
        X_saltpepper_5 = generate_noise(X_hat, img_shape, noise_type="saltpepper", level=0.05, ratio=0.5)
        X_saltpepper_10 = generate_noise(X_hat, img_shape, noise_type="saltpepper", level=0.1, ratio=0.5)
        X_saltpepper_20 = generate_noise(X_hat, img_shape, noise_type="saltpepper", level=0.2, ratio=0.5)

        # Create 3 gaussian noise images
        X_gaussian_01 = generate_noise(X_hat, img_shape, noise_type="gaussian", mean=0, std=0.01)
        X_gaussian_05 = generate_noise(X_hat, img_shape, noise_type="gaussian", mean=0, std=0.05)
        X_gaussian_10 = generate_noise(X_hat, img_shape, noise_type="gaussian", mean=0, std=0.1)

        # put the original and contaminated data in a dict
        X_hat_noisy_maplist.append({
        None: X_hat,
        "block_04": X_block_04,
        "saltpepper_5": X_saltpepper_5,
        "gaussian_01": X_gaussian_01,
        "block_08": X_block_08,
        "saltpepper_10": X_saltpepper_10,
        "gaussian_05": X_gaussian_05,
        "block_12": X_block_12,
        "saltpepper_20": X_saltpepper_20,
        "gaussian_10": X_gaussian_10
        })

    return X_hat_noisy_maplist

def NMF_and_result(nmf="l2", img_shape = None, noise=None,noise_mapping = None, K=50, max_iter=100, tol=1e-4, display = False):
    """
    Create a list that include maps from noise type to generated noisy images

    Parameters:
    - nmf: The type of nmf, one of l2, l1, CIM and huber
    - img_shape: The shape of image
    - noise: The type of noise
    - noise_mapping: The map that contents original and contaminated data
    - K: The number of component
    - max_iter: The maximum of iteration times
    - tol: The tolorance of convergence
    - display: To decide whether to visualize
    
    """
    function_mapping = {
        "l2": nmf_frobenius,
        "l1": l1_norm_nmf,
        "CIM": cim_nmf,
        "huber": huber_nmf
    }
    if nmf in function_mapping and noise in noise_mapping:
        D, R = function_mapping[nmf](noise_mapping[noise], K=K, max_iter=max_iter, tol=tol)
    else:
        print("Invalid nmf or noise value.")
        return None, None, None

    # print("D.shape = {}, R.shape = {}".format(D.shape, R.shape))
    RMSE = sqrt(mean_squared_error(noise_mapping[None], D.dot(R)))
    # print noise infomation
    if display:
        # Display the original and reconstructed images side by side.
        if noise is not None:
            print("="*40,"Noise: {}".format(noise),"="*40)
        else:
            print("="*40,"No noise","="*40)
        
        display_reconstructed_images(noise_mapping[None],Title= "Original Image", img_shape=img_shape)
        if noise != None:
            display_reconstructed_images(noise_mapping[noise], Title = "Noisy Image", img_shape=img_shape)

        display_reconstructed_images(D.dot(R), Title = "Reconstructed", img_shape=img_shape)

    return D, R, RMSE



def compute_result(allresultmap, Y_hat_list):
    """
    Calculate the result and demostrate as a table

    Parameters:
    - allresultmap: All of the predicted result from all kinds of algrithm and noise
    - Y_hat_list: The list of true lable of validation data
    
    """
    
    all_avg_rmse, all_std_rmse, all_avg_acc, all_std_acc, all_avg_nmi, all_std_nmi = [],[],[],[],[],[]

    for resultmap in allresultmap: # for L1, L2, CIM and Huber results
        rmse, avg_rmse, std_rmse = [], [], []
        acc, avg_acc, std_acc = [], [], []
        nmi, avg_nmi, std_nmi = [], [], []
        k = list(resultmap.keys())
        fi,si = 0,10
        for j in range(3):
            for i in k[fi:si]:
                    a, b = calculate_acc_nmi(resultmap[i][1], Y_hat_list[j])
                    rmse.append(resultmap[i][2])
                    acc.append(a)
                    nmi.append(b)
            fi +=10
            si +=10

        index_list = [np.arange(i ,30,10) for i in range(10)]
        for index in index_list:
            a, b = np.mean([rmse[i] for i in index]), np.std([rmse[i] for i in index])
            avg_rmse.append(a)
            std_rmse.append(b)

            a, b = np.mean([acc[i] for i in index]), np.std([acc[i] for i in index])
            avg_acc.append(a)
            std_acc.append(b)

            a, b = np.mean([nmi[i] for i in index]), np.std([nmi[i] for i in index])
            avg_nmi.append(a)
            std_nmi.append(b)
        all_avg_rmse.append(avg_rmse)
        all_std_rmse.append(std_rmse)
        all_avg_acc.append(avg_acc)
        all_std_acc.append(std_acc)
        all_avg_nmi.append(avg_nmi)
        all_std_nmi.append(std_nmi)

    
    # print the RMSE in a table like acc and nmi table
    headers_rmse = ["Algorithm","Noise", "RMSE-mean", "RMSE-std",'Accuracy-mean','Accuracy-std','NMI-mean','NMI-std']

    # display the result with a table
    for i,j in zip(range(4), ['L1-NMF','L2-NMF','CIM-NMF','Huber-NMF']):
        rmse_table = [
            [j ,"No noise",  all_avg_rmse[i][0], all_std_rmse[i][0], all_avg_acc[i][0], all_std_acc[i][0],all_avg_nmi[i][0], all_std_nmi[i][0]],
            ["","Block_04",  all_avg_rmse[i][1], all_std_rmse[i][1], all_avg_acc[i][1], all_std_acc[i][1],all_avg_nmi[i][1], all_std_nmi[i][1]],
            ["","Block_08",  all_avg_rmse[i][2], all_std_rmse[i][2], all_avg_acc[i][2], all_std_acc[i][2],all_avg_nmi[i][2], all_std_nmi[i][2]],
            ["","Block_12",  all_avg_rmse[i][3], all_std_rmse[i][3], all_avg_acc[i][3], all_std_acc[i][3],all_avg_nmi[i][3], all_std_nmi[i][3]],
            ["","Saltpepper_5", all_avg_rmse[i][4], all_std_rmse[i][4], all_avg_acc[i][4], all_std_acc[i][4],all_avg_nmi[i][4], all_std_nmi[i][4]],
            ["","Saltpepper_10", all_avg_rmse[i][5], all_std_rmse[i][5], all_avg_acc[i][5], all_std_acc[i][5],all_avg_nmi[i][5], all_std_nmi[i][5]],
            ["","Saltpepper_20", all_avg_rmse[i][6], all_std_rmse[i][6], all_avg_acc[i][6], all_std_acc[i][6],all_avg_nmi[i][6], all_std_nmi[i][6]],
            ["","Gaussian_01", all_avg_rmse[i][7], all_std_rmse[i][7], all_avg_acc[i][7], all_std_acc[i][7],all_avg_nmi[i][7], all_std_nmi[i][7]],
            ["","Gaussian_05", all_avg_rmse[i][8], all_std_rmse[i][8], all_avg_acc[i][8], all_std_acc[i][8],all_avg_nmi[i][8], all_std_nmi[i][8]],
            ["","Gaussian_10", all_avg_rmse[i][9], all_std_rmse[i][9], all_avg_acc[i][9], all_std_acc[i][9],all_avg_nmi[i][9], all_std_nmi[i][9]] 
            

        ]

        print(tabulate(rmse_table, headers=headers_rmse, tablefmt="github"))
