import numpy as np
import cv2
import math
import time
   
def get_features(image, x, y, feature_width, scales=None):

    keypoints =[(x[i],y[i]) for i in range(len(x))]
   
    ####################################################################
    ## One way
    # kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # def apply_filter(image, kernel):
    #     return cv2.filter2D(image, -1, kernel)


    # I_x = apply_filter(image, kernel_x)
    # I_y = apply_filter(image, kernel_y)
    ###################################################################

    I_x = cv2.Sobel(image,cv2.CV_64F, 1,0, 3)
    I_y = cv2.Sobel(image,cv2.CV_64F, 0,1, 3)
    
    ## Magnitude matrix
    grad_mag_mat = np.sqrt(np.square(I_x) + np.square(I_y))
    
    ## Orientation matrix
    grad_ang_mat = np.rad2deg(np.arctan2(I_y, I_x))
    
	## Gaussian applied on magnitude matrix
    gaussian_on_mag = cv2.GaussianBlur(grad_mag_mat, (3, 3), sigmaX=1)

	## Extract big grid
	
    def box_around_point(image, location, feature_width = 16):

        window = feature_width//2
        m1 = np.subtract(location , [window-1, window-1]).tolist()
        m2 = np.add(location , [window, window]).tolist()
        
        sub_matrix = image[m1[0]:m2[0]+1, m1[1]:m2[1]+1]

        return np.array(sub_matrix)
	
    ## Extract small grids
    def extract_small_boxes(matrix, size = 4):

        sub_matrices= []
        loop_len = int(matrix.shape[0] / size)
        for i in range(loop_len):
            for j in range(loop_len):
                sub_matrix = matrix[i:i+size ,j:j+size]
                sub_matrices.append((sub_matrix,(i,j),(i+size,j+size)))
        return sub_matrices    



    # def histogram(matrix, weights, bin_no):

    #     bin_width = 360 // bin_no

    #     temp = matrix.flatten()
    #     print(len(temp))
    #     print(len(weights))
    #     for i in range(len(temp)):
    #         if temp[i] < 0 :
    #             temp[i]  = 360 + temp[i]
                
    #     hist = np.histogram(temp, bins=[i for i in range(0,361,bin_width)], weights = weights)[0]
    #     return hist
    
    ## Adjust angle
    
    def adjust_angle(ang_matrix):

        for i in range(ang_matrix.shape[0]):
            for j in range(ang_matrix.shape[1]):
                if ang_matrix[i,j] < 0 :
                    ang_matrix[i,j]  = 360 + ang_matrix[i,j]
        return ang_matrix

	## Histogram, Interpolation
	
    def interpolated_hist(matrix, weight_matrix, bin_no):
        
        #print("Building Histogram with bin:", bin_no)
        bin_width = 360//bin_no
        matrix = adjust_angle(matrix)

        mat_vals = matrix.flatten()
        weights = weight_matrix.flatten()

                
        mid_points = [i*bin_width + bin_width/2 for i in range(bin_no)]       
        weighted_counts = [0 for i in range(bin_no)]

        for i in range(len(mat_vals)):
            for j in range(bin_no):
        
                if mat_vals[i] <= mid_points[j]:

                    temp = 1 - ((mid_points[j] - mat_vals[i])/bin_width)
                   
                    weighted_counts[j] += weights[i] * temp
                    weighted_counts[j-1] += weights[i] * (1-temp)
                    break
                
                elif j == bin_no -1 and mat_vals[i] > mid_points[j]:
                    temp = 1 - (mat_vals[i] - mid_points[j])/bin_width
                    weighted_counts[j] += weights[i] * temp
                    weighted_counts[0] += weights[i] * (1-temp)
                    break
        #print(weighted_counts)
        return np.array(weighted_counts)
    
    ## Return the final sift vector
    def return_sift_vector(sub_matrix, weight_matrix, feature_width):

        grids = extract_small_boxes(sub_matrix, 4)
       
        feature_vectors = []
        
        for i in range(len(grids)):
            m1 = grids[i][1]
            m2 = grids[i][2]
            
            weight_mat = weight_matrix[m1[0]:m2[0]+1, m1[1]:m2[1]+1]
           
            #hist_2 = list(histogram(grids[i][0],weights,8))
            hist = interpolated_hist(grids[i][0], weight_mat, 8 )
            
            if sum(hist) != 0:
                normalized_hist = hist / sum(hist)
            else:
                normalized_hist = hist    
            feature_vectors.append(normalized_hist)
            
        sift_vector = (np.array(feature_vectors)).flatten()

    
        return sift_vector

    ## Fit parabola

    def fit_parabola(hist):

        bin_width = 360 // len(hist)

        dom_bin_nos = hist.argsort()[-3:][::-1]
        dominant_ang = [temp*bin_width + bin_width/2 for temp in dom_bin_nos]
        
        sorted_hist = hist.copy()
        sorted_hist.sort()
        
        dom_vals = sorted_hist[-3:][::-1]

        if dom_vals[1] > 0.8 * dom_vals[0] and dom_vals[2] > 0.8 * dom_vals[0] :
            pass
        else: 
            return(dominant_ang[0])    
        
        # ax^2 + b*x + c = y
        A= []
        for i in dominant_ang:
            A.append([i**2,i,1])
        A = np.array(A)
        b = np.array(dom_vals)
        x = np.linalg.solve(A, b)

        return -x[1]/(2*x[0])

    ## To make rotation invariant
    def rotation_invariant(matrix,key_point):

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if (i,j) != key_point:
                    matrix[i,j] = matrix[i,j] - matrix[key_point]
                else:
                    pass
        return matrix       

    feature_vectors = []

    ## All the steps sequentially

    for point in keypoints:

        temp_ang_mat = grad_ang_mat.copy()
        matrix = box_around_point(grad_ang_mat, point, feature_width)
        matrix = adjust_angle(matrix)

        weight_matrix = box_around_point(gaussian_on_mag, point, feature_width)
        
        hist = interpolated_hist(matrix, weight_matrix, 36)

        new_ang = fit_parabola(hist)
        temp_ang_mat[point] = new_ang
        temp_ang_mat = rotation_invariant(temp_ang_mat, point)

        feature_matrix= box_around_point(temp_ang_mat,point, feature_width)

        feature_matrix = adjust_angle(feature_matrix)
        sift_vector = return_sift_vector(feature_matrix, weight_matrix,feature_width)

        feature_vectors.append(sift_vector)
    

    return np.array(feature_vectors)
    















    # return fv
