'''
ECE276A WI19 HW1
Blue Barrel Detector
'''
import numpy as np
import os, cv2
from skimage.measure import label, regionprops

class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.blue_mean = np.array([[115.00799798], [ 65.94341023], [ 32.35539942]])
        self.not_blue_mean = np.array([[ 87.11597537], [ 98.0918099 ], [104.67259641]])

        self.blue_cov = np.array([[3219.87199431, 1866.08337731, 573.04527479], [1866.08337731, 1481.42775646,753.69162267], [ 573.04527479,  753.69162267,  875.6160161 ]])
        self.not_blue_cov = np.array([[3834.88244176, 3583.18734159, 3463.34380916], [3583.18734159, 3548.6991964, 3496.48462157],
 [3463.34380916, 3496.48462157, 3687.20627384]])
        
        self.blue_cov_inv = np.linalg.inv(self.blue_cov)
        self.not_blue_cov_inv = np.linalg.inv(self.not_blue_cov)

        self.blue_cov_det = np.linalg.det(self.blue_cov)
        self.not_blue_cov_det = np.linalg.det(self.not_blue_cov)
    
    
        self.blue_prior = 0.009013078703703704
        self.not_blue_prior = 0.9872857870370371

        
    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        # YOUR CODE HERE
        array_img = np.array(img)  

        my_mask =np.zeros((array_img.shape[0], array_img.shape[1]), dtype=np.uint8)

        for i in range(array_img.shape[0]):
            for j in range(array_img.shape[1]):
                current_pixel = array_img[i,j,:]
                current_pixel = np.reshape(current_pixel, (3,1))
                blue_prob = multi_gauss(current_pixel, self.blue_mean, self.blue_cov_inv, self.blue_cov_det, 3)*self.blue_prior
                not_blue_prob = multi_gauss(current_pixel, self.not_blue_mean, self.not_blue_cov_inv, self.not_blue_cov_det, 3)*self.not_blue_prior

                if blue_prob > not_blue_prob:
                    my_mask[i,j] = 1
                      
        return my_mask

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        my_mask = self.segment_image(img)
             
        #post segmentation processing
        kernel = np.ones((5,5),np.uint8)
        my_mask = cv2.morphologyEx(my_mask,cv2.MORPH_OPEN, kernel)
        my_mask = cv2.morphologyEx(my_mask, cv2.MORPH_CLOSE, kernel)
        my_mask = cv2.dilate(my_mask,kernel,iterations = 5)
        my_mask = cv2.erode(my_mask,kernel,iterations = 5)
                
        #remove shapes with small area
        cc, hierarchy = cv2.findContours(my_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)      
        for c in cc:
            if cv2.contourArea(c) < 500:
                my_mask = cv2.fillPoly(my_mask, pts = [c], color=(0,0,0))

        #check aspect ratio
        cc, hierarchy = cv2.findContours(my_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        for c in cc:       
            x,y,w,h = cv2.boundingRect(c)
            aspect_ratio_1 = float(w)/h
            aspect_ratio_2 = float(h)/w
            if aspect_ratio_1 > 1 or aspect_ratio_2 > 3:
                my_mask = cv2.fillPoly(my_mask, pts =[c], color=(0,0,0))
              
        boxes = []
        cc, hierarchy = cv2.findContours(my_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        for c in cc:              
            x,y,w,h = cv2.boundingRect(c)
            boxes.append([x,y,x+w,y+h])

        return boxes
    
def multi_gauss(x, mean, cov_inv, cov_det, dim):
    '''
    Takes an array of one input and calculates its probability based on the Guassians parameters
    
    Arguments:
    x(ndarray): single input to be plugged into pdf
    mean(ndarray): mean of the Gaussian
    cov(ndarray): covariance of the Gaussian
    dim(int): dimension of the Gaussian
    
    Returns:
    probs(ndarray): probabilities for each input
    '''
    
    assert isinstance(x, np.ndarray), "x is not a numpy array"
    assert isinstance(mean, np.ndarray), "mean is not a numpy array"
    assert isinstance(cov_inv, np.ndarray), "cov is not a numpy array"
    assert isinstance(dim, int), "dim is not a int"
    assert x.shape == (3,1)
    assert mean.shape == (3,1)
    assert cov_inv.shape == (3,3)
    
    
    a = 1/np.sqrt( ((2*np.pi)**dim) * cov_det)
    
    minus_mean = x - mean
    exp_term = -0.5*np.dot( np.dot(minus_mean.T, cov_inv), minus_mean)
    exp = np.exp(exp_term)
    
    probs = a*exp
    
    return probs[0][0]


if __name__ == '__main__':
    #Parameters obtained during training    
    folder = "trainset"
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Display results:
        #(1) Segmented images
        mask_img = my_detector.segment_image(img)
        #(2) Barrel bounding box
        boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope


