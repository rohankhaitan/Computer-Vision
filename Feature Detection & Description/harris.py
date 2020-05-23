import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import time
import operator
#import itertools
#from tqdm import tqdm as tq


def get_interest_points(image, feature_width, window =1, ANMS= True):

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    def apply_filter(image, kernel):
        return cv2.filter2D(image, -1, kernel)

    I_x = apply_filter(image, kernel_x)
    I_y = apply_filter(image, kernel_y)

    Ixx = cv2.GaussianBlur(I_x**2, (5, 5), sigmaX=1)
    Ixy = cv2.GaussianBlur(I_y*I_x, (5, 5), sigmaX=1)
    Iyy = cv2.GaussianBlur(I_y**2, (5, 5), sigmaX=1)

    # ##Another way
    # I_x = cv2.Sobel(image,cv2.CV_64F, 1,0, 3)
    # I_y = cv2.Sobel(image,cv2.CV_64F, 0,1, 3)
    
    ## k can be taken between 0.04 to 0.06
    k = 0.06

    # determinant
    detA = Ixx * Iyy - Ixy ** 2

    # trace
    traceA = Ixx + Iyy
    
    # harris resonse
    harris_response = detA - k * traceA ** 2


########################################################################################


    # def difference_of_list(t1, t2):
    #     output = [t1[i] - t2[i] for i in range(len(t1))]
    #     return tuple(output)

    # def find_neighbour_locations(location, window):
    #     values = [i for i in range(-window, window+1)]
    #     all_perm = [p for p in itertools.product(values, repeat=2)]
    #     all_perm.remove((0, 0))
    #     locations_to_compare = [difference_of_list(location, i) for i in all_perm]
    #     return locations_to_compare
    #     #return all_perm

    # def compare(image, location, window = 1):
    #     neighbour_locations = find_neighbour_locations(location, window)
    #     temp = []
    #     flag = True
    #     for k in neighbour_locations:
    #         i, j = k
    #         if i not in range(0, image.shape[0]) or j not in range(0, image.shape[1]):
    #             pass
    #         else:
    #             if image[k] >= image[location]:
    #                 flag = False
    #                 break

    #     return flag
    
#########################################################################################

    def keep_range_in_bound(co_ordinate, feature_width = 16):
    
        if co_ordinate[0]<0:
            co_ordinate[0]=0
        elif co_ordinate[0]>=image.shape[0]:
            co_ordinate[0]= image.shape[0]-1
        
        if co_ordinate[1]<0:
            co_ordinate[1]=0
        elif co_ordinate[1]>=image.shape[1]:
            co_ordinate[1]= image.shape[1]-1
        return co_ordinate


    ## Check for local maxima

    def compare(image,location,window ):

        m1 = np.subtract(location , [window, window]).tolist()
        m2 = np.add(location , [window, window]).tolist()

        #all_loc = find_neighbour_locations(location, window)
        #m1, m2 = min(all_loc), max(all_loc)
        m1 , m2 = keep_range_in_bound(list(m1)), keep_range_in_bound(list(m2))
        sub_matrix = image[m1[0]:m2[0]+1, m1[1]:m2[1]+1]
        #return sub_matrix
    
        temp = sub_matrix.flatten()
        max_val = np.max(np.delete(temp, len(temp)//2))
        if image[location] > max_val:
            return True
        else:
            return False
    
    ## Find the interest points

    def find_interest_points(image, window):
        r_max = np.max(image)
        interest_points = []
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                location = (i, j)
                if image[location] > 0.009 * r_max:
                    if compare(image, location,window):
                        interest_points.append((i,j))
        return interest_points


    interest_points = find_interest_points(harris_response, window)
    print("No of interest Points:", len(interest_points))

    # x = np.array([i[0] for i in interest_points])
    # y = np.array([i[1] for i in interest_points])
	
	## Interest points based on feature width 
	
    def check_point(image, point, feature_width):

        fw = feature_width/ 2
        m = image.shape[0]
        n = image.shape[1]
        x,y = point
        if (x - fw) < 0 or (x+fw) >= m or (y-fw) < 0 or (y+fw)>= n:
            return False
        else:
            return True    
	
	## Find the interest points
	
    final_interest_points = []
    for point in interest_points:
        if check_point(harris_response,point, feature_width):
            final_interest_points.append(point)
        else:
            pass    
    
    print("No of interest Points based on feature width:", len(final_interest_points))
	
	## ANMS
	
    def do_ANMS(interest_points, harris_response):

        main_list = []
        for k in interest_points:
            i, j = k
            main_list.append((i, j, harris_response[k]))

        main_list.sort(key=operator.itemgetter(2), reverse=True)

        def find_e_dist(tup, l):
            i, j, k = tup
            min_val = 999999
            for val in l:
                if val[2] > 1.1 * tup[2]:
                    temp = np.linalg.norm([i-val[0], j-val[1]])
                    if temp < min_val:
                        min_val = temp
            return (i, j, k, min_val)

        adaptive_list = []
        for i in range(len(main_list)):
            tup = main_list[i]
            adaptive_list.append(find_e_dist(tup, main_list[:i]))

        #print("Length of adaptive list:", len(interest_points))

        adaptive_list.sort(key=operator.itemgetter(3), reverse=True)
        return adaptive_list

    if ANMS == True:
        
        adaptive_list = do_ANMS(final_interest_points, harris_response)
        x = np.array([i[0] for i in adaptive_list])
        y = np.array([i[1] for i in adaptive_list])
        print("ANMS done..  Can choose the number of points.")
        return ((x,y))
    
    else:
        x = np.array([i[0] for i in final_interest_points])
        y = np.array([i[1] for i in final_interest_points])
        print("Result without doing ANMS ..")
        return ((x,y))

