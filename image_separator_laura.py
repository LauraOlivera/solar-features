import numpy as np
import matplotlib.pyplot as plt
import astropy
from matplotlib.pyplot import cm
from astropy import constants as const
from astropy import units as u
import numpy.ma as ma
from datetime import datetime
from scipy.stats import mode
from scipy.misc import imread
import imageio
import matplotlib.patches as patches
from matplotlib.patches import Circle
from skimage import img_as_float, measure, feature
from skimage.transform import rotate as skrotate
from photutils.centroids import centroid_com
from itertools import combinations
import os
from datetime import timedelta  

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import time
import matplotlib.patches as mpatches
import cv2
import old_code as old


class sun_img(object):
	def __init__(self,path):
		self.path=path
		self.original = cv2.imread(self.path)
		#self.fullimage=img_as_float(imageio.imread(path))
		#self.image=self.crop()
		#self.time=os.path.getmtime(self.path)
		#self.totintensity=self.intensity()


	def separate_images(self):
		#original = cv2.imread(self.path)
		original = self.original
		original.astype('uint8')
		# Set the image in grayscale
		gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	
		# Filter for better circle detection
		#print('Warning: the threshold is set randomly, check this')
		filtered =cv2.threshold(original,80,0,type=cv2.THRESH_TOZERO)
		circle_output = filtered[1].copy()
		#filtered_gray = cv2.cvtColor(filtered[1], cv2.COLOR_BGR2GRAY)
		# detect circles in the image
		#circles = cv2.HoughCircles(filtered_gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
		#circles = cv2.HoughCircles(filtered_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
		#circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

		 
		# ensure at least some circles were found
		if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
			circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		# This is for plotting only
		#for (x, y, r) in circles2:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			#cv2.circle(circle_output, (x, y), r, (0, 255, 0), 1)
			#cv2.rectangle(circle_ouput, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		
		# Now we obtain the image of the inside

		height,width,depth = original.shape
		circle_img = np.zeros((height,width), np.uint8)
		cv2.circle(circle_img,(circles[0][0], circles[0][1]),circles[0][2]-1,1,thickness=-1)
		inside_image1 = cv2.bitwise_and(original, original, mask=circle_img)
		outside_image = cv2.bitwise_xor(original, inside_image1)
		circle_img2 = np.zeros((height,width), np.uint8)
		cv2.circle(circle_img2,(circles[0][0], circles[0][1]),circles[0][2],1,thickness=-1)
		inside_image = cv2.bitwise_and(original, original, mask=circle_img2)
		return outside_image, inside_image

	def circle_properties(self):
		original = cv2.imread(self.path)
		original.astype('uint8')
		# Set the image in grayscale
		gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	
		# Filter for better circle detection
		#print('Warning: the threshold is set randomly, check this')
		filtered =cv2.threshold(original,80,0,type=cv2.THRESH_TOZERO)
		circle_output = filtered[1].copy()
		#filtered_gray = cv2.cvtColor(filtered[1], cv2.COLOR_BGR2GRAY)
		# detect circles in the image
		#circles = cv2.HoughCircles(filtered_gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
		#circles = cv2.HoughCircles(filtered_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
		#circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

		 
		# ensure at least some circles were found
		if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
			circles = np.round(circles[0, :]).astype("int")

		return circles




	def intensity(img):
	    # img must not be a gray image
		#img = cv2.imread(self.path)
		temp=[img[j][i][0] for j in range(img.shape[0])  for i in range(img.shape[1])]
		intensity = np.reshape(temp, [img.shape[0],img.shape[1]])
		return intensity

	def intensity_gray(img):
		# img must be a gray image
		#img1 = cv2.imread(self.path)
		#img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		temp=[img[j][i] for j in range(img.shape[0])  for i in range(img.shape[1])]
		intensity = np.reshape(temp, [img.shape[0],img.shape[1]])
		return intensity

	def detecting_filaments(self):
		# First we load the image
		test = cv2.imread(self.path)
		# Then we split the image to keep the inside using separate_images
		a, b = sun_img.separate_images(self)
		
		# Compute radius and center of the Sun
		r = sun_img.circle_properties(self)[0][2]
		x = sun_img.circle_properties(self)[0][0]
		y = sun_img.circle_properties(self)[0][1]

		center = (x,y)

		# Compute intensities
		mean_inten = np.mean(sun_img.intensity(b)[np.nonzero(sun_img.intensity(b))])
		median_inten = np.median(sun_img.intensity(b)[np.nonzero(sun_img.intensity(b))])

		# Define effective intensity
		mean_int=1.266*mean_inten

		# Make gray images

		b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
		b_gray1 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

		# Replace the values above the effective mean with the median 
		b_gray[np.where(b_gray1 > mean_int)]=median_inten


		# Use edge detector to identify possible ROI

		edges = cv2.Canny(b_gray,mean_int-100,mean_int-10)




		# Remove the circumference of the sun from the detected edges using a mask of a slightly smaller (r-7) radius
		maskk = np.zeros(edges.shape, dtype=np.uint8)
		cv2.circle(maskk, (148,151), r-7, (255, 255, 255), -1, 8, 0)
		max_loc = (148,151)
		circle_radius = 132
		# Apply mask (using bitwise & operator)
		result_array = cv2.bitwise_and(edges, maskk, mask = maskk)

		
		# Now in order to connect the inside of the ROI we apply a procedure of erosion and subsequent dilation 
		
		kernel = np.ones((5,5),np.uint8)
		# Dilation with a fat kernel
		dilation = cv2.dilate(result_array,kernel,iterations = 1)

		# Erosion to recover the original shape
		erosion = cv2.erode(dilation,kernel,iterations = 1)
		
		# Now we want to tag the structures that have a surface bigger than a threshold to only keep filaments and discard spots


		image = erosion.copy()
		# apply threshold
		thresh = threshold_otsu(image)
		bw = closing(image > thresh, square(3))

		# remove artifacts connected to image border
		cleared = clear_border(bw)

		# label image regions
		label_image = label(cleared)
		image_label_overlay = label2rgb(label_image, image=image)


		# Interesting will be the rectangular segments of the original gray image that contain a feature
		interesting = []
		# Features as selected by the surface criteria
		regions = []
		for region in regionprops(label_image, intensity_image=b_gray):
			# take regions with large enough areas
			if region.area >= 25:
				# Make a list with the regions that make the cut
				minr, minc, maxr, maxc = region.bbox
				roi = b_gray[minr:minr+maxr - minr,minc:minc+maxc - minc]
				interesting.append(roi)
				regions.append(region)





		# remove the filaments from the background in the interesting boxes

		empty_b_gray = b_gray.copy()
		mask = np.zeros(np.shape(empty_b_gray))    

		# For each of the regions in the list, we define the coordinates of said region, x_c and y_c and create a mask in those 		coordinates
		for k in regions:

			x_c = np.zeros(len(k.coords))
			y_c = np.zeros(len(k.coords))

			for i in np.arange(len(k.coords)):
				x_c[i] = k.coords[i][0]
				y_c[i] = k.coords[i][1]

				mask[x_c.astype(int),y_c.astype(int)]=True
		
		# Now we use that mask to create a gray image with the pixels of the features missing
		empty_b_gray = ma.masked_where(mask, empty_b_gray)


		# Now we want to create the rectangular segments around the empty features
		# Empty interesting and empty regions are equivalent to before but wrt the masked image

		empty_interesting = []
		empty_regions = []

		for region in regionprops(label_image, intensity_image=empty_b_gray):
			# take regions with large enough areas
			if region.area >= 25:
				# draw rectangle around segmented coins
				minr, minc, maxr, maxc = region.bbox
				empty_roi = empty_b_gray[minr:minr+maxr - minr,minc:minc+maxc - minc]
				empty_interesting.append(empty_roi)

		# Now we compute the intensities of the candidate features
		intensities = np.zeros(len(regions))
		for i in np.arange(len(regions)):
			intensities[i]=regions[i].mean_intensity

		
		# Now we compute the intensities of the background in the boxes after substracting the feature
		bg_mean = []

		for i in empty_interesting:
			m = np.mean(i)
			bg_mean.append(m)


		# defining the final candidates using the background mean intensity as reference value and creating a list of possible 			filaments
		possible_filaments=[]
		for i in np.arange(len(interesting)):
			if bg_mean[i]>intensities[i]:
				possible_filaments.append((regions[i],interesting[i]))

		# Use preliminar method to discriminate between filaments and other regions: only indicative

		final_classification = []

		#trying to select filaments basing on the ratio of the lengths of the sides of the boxes of the final candidates 
		for i in np.arange(len(possible_filaments)):
			h_box = possible_filaments[i][0].bbox[2]-possible_filaments[i][0].bbox[0]
			v_box = possible_filaments[i][0].bbox[3]-possible_filaments[i][0].bbox[1]
			ratio = possible_filaments[i][0].extent
			# We label the most probable filaments with a 1 and less probable with a 0
			if (h_box/v_box < 1.2) and (h_box/v_box > 0.8) and (ratio >0.33):
				final_classification.append([possible_filaments[i][0],possible_filaments[i][1],0])
			else:
				final_classification.append([possible_filaments[i][0],possible_filaments[i][1],1])

		results = np.empty((len(final_classification)+1,7), dtype=object)
		for i in np.arange(len(final_classification)):
			# Table header
			results[0,:]= ['Name', 'Center Coord.', 'Box', 'Intensity', 'Area', 'Likely filament?', 'Original image']
			# Name of the filament: 1, 2, 3, 4...
			results[i+1,0]=str(i+1)

			# Coordinates of center
			results[i+1,1]= final_classification[i][0].centroid

			# Dimensions of box enclosing the region: 
			# Bounding box ``(min_row, min_col, max_row, max_col)``.
			# Pixels belonging to the bounding box are in the half-open interval
			# ``[min_row; max_row)`` and ``[min_col; max_col)``.
			results[i+1,2] = final_classification[i][0].bbox

			# Intensity of the region
			results[i+1,3] = final_classification[i][0].mean_intensity

			# Area of the region
			results[i+1,4] = final_classification[i][0].area

			# Likely a filament or not
			if final_classification[i][2]:
				results[i+1,5] = 'Probably yes'
			else:
				results[i+1,5] = 'Probably not'
			
			# From what image it comes 
			results[i+1,6] = self.path

		# Plot the final candidates in the original image. Red box means probably not, green probably yes.

		possible_regions = [i[0] for i in final_classification]
		marker = [i[2] for i in final_classification]

		fig1, ax = plt.subplots(figsize=(10, 6))
		ax.imshow(b)
		for i in np.arange(len(final_classification)):
			if marker[i] == 0:
				minr, minc, maxr, maxc = possible_regions[i].bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
							  fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)
				plt.text(maxc+1, maxr,str(i+1))

			else:
				minr, minc, maxr, maxc = possible_regions[i].bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
							  fill=False, edgecolor='green', linewidth=2)
				ax.add_patch(rect)
				plt.text(maxc+1, maxr,str(i+1))


		ax.set_axis_off()
		plt.tight_layout()
		plt.show()

		return results

	def detecting_filaments_no_plot(self):
		# First we load the image
		test = cv2.imread(self.path)
		# Then we split the image to keep the inside using separate_images
		a, b = sun_img.separate_images(self)
		
		# Compute radius and center of the Sun
		r = sun_img.circle_properties(self)[0][2]
		x = sun_img.circle_properties(self)[0][0]
		y = sun_img.circle_properties(self)[0][1]

		center = (x,y)

		# Compute intensities
		mean_inten = np.mean(sun_img.intensity(b)[np.nonzero(sun_img.intensity(b))])
		median_inten = np.median(sun_img.intensity(b)[np.nonzero(sun_img.intensity(b))])

		# Define effective intensity
		mean_int=1.266*mean_inten

		# Make gray images

		b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
		b_gray1 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

		# Replace the values above the effective mean with the median 
		b_gray[np.where(b_gray1 > mean_int)]=median_inten


		# Use edge detector to identify possible ROI

		edges = cv2.Canny(b_gray,mean_int-100,mean_int-10)




		# Remove the circumference of the sun from the detected edges using a mask of a slightly smaller (r-7) radius
		maskk = np.zeros(edges.shape, dtype=np.uint8)
		cv2.circle(maskk, (148,151), r-7, (255, 255, 255), -1, 8, 0)
		max_loc = (148,151)
		circle_radius = 132
		# Apply mask (using bitwise & operator)
		result_array = cv2.bitwise_and(edges, maskk, mask = maskk)

		
		# Now in order to connect the inside of the ROI we apply a procedure of erosion and subsequent dilation 
		
		kernel = np.ones((5,5),np.uint8)
		# Dilation with a fat kernel
		dilation = cv2.dilate(result_array,kernel,iterations = 1)

		# Erosion to recover the original shape
		erosion = cv2.erode(dilation,kernel,iterations = 1)
		
		# Now we want to tag the structures that have a surface bigger than a threshold to only keep filaments and discard spots


		image = erosion.copy()
		# apply threshold
		thresh = threshold_otsu(image)
		bw = closing(image > thresh, square(3))

		# remove artifacts connected to image border
		cleared = clear_border(bw)

		# label image regions
		label_image = label(cleared)
		image_label_overlay = label2rgb(label_image, image=image)


		# Interesting will be the rectangular segments of the original gray image that contain a feature
		interesting = []
		# Features as selected by the surface criteria
		regions = []
		for region in regionprops(label_image, intensity_image=b_gray):
			# take regions with large enough areas
			if region.area >= 25:
				# Make a list with the regions that make the cut
				minr, minc, maxr, maxc = region.bbox
				roi = b_gray[minr:minr+maxr - minr,minc:minc+maxc - minc]
				interesting.append(roi)
				regions.append(region)





		# remove the filaments from the background in the interesting boxes

		empty_b_gray = b_gray.copy()
		mask = np.zeros(np.shape(empty_b_gray))    

		# For each of the regions in the list, we define the coordinates of said region, x_c and y_c and create a mask in those coordinates
		for k in regions:

			x_c = np.zeros(len(k.coords))
			y_c = np.zeros(len(k.coords))

			for i in np.arange(len(k.coords)):
				x_c[i] = k.coords[i][0]
				y_c[i] = k.coords[i][1]

				mask[x_c.astype(int),y_c.astype(int)]=True
		
		# Now we use that mask to create a gray image with the pixels of the features missing
		empty_b_gray = ma.masked_where(mask, empty_b_gray)


		# Now we want to create the rectangular segments around the empty features
		# Empty interesting and empty regions are equivalent to before but wrt the masked image

		empty_interesting = []
		empty_regions = []

		for region in regionprops(label_image, intensity_image=empty_b_gray):
			# take regions with large enough areas
			if region.area >= 25:
				# draw rectangle around segmented coins
				minr, minc, maxr, maxc = region.bbox
				empty_roi = empty_b_gray[minr:minr+maxr - minr,minc:minc+maxc - minc]
				empty_interesting.append(empty_roi)

		# Now we compute the intensities of the candidate features
		intensities = np.zeros(len(regions))
		for i in np.arange(len(regions)):
			intensities[i]=regions[i].mean_intensity

		
		# Now we compute the intensities of the background in the boxes after substracting the feature
		bg_mean = []

		for i in empty_interesting:
			m = np.mean(i)
			bg_mean.append(m)


		# defining the final candidates using the background mean intensity as reference value and creating a list of possible filaments
		possible_filaments=[]
		for i in np.arange(len(interesting)):
			if bg_mean[i]>intensities[i]:
				possible_filaments.append((regions[i],interesting[i]))

		# Use preliminar method to discriminate between filaments and other regions: only indicative

		final_classification = []

		#trying to select filaments basing on the ratio of the lengths of the sides of the boxes of the final candidates 
		for i in np.arange(len(possible_filaments)):
			h_box = possible_filaments[i][0].bbox[2]-possible_filaments[i][0].bbox[0]
			v_box = possible_filaments[i][0].bbox[3]-possible_filaments[i][0].bbox[1]
			ratio = possible_filaments[i][0].extent
			# We label the most probable filaments with a 1 and less probable with a 0
			if (h_box/v_box < 1.2) and (h_box/v_box > 0.8) and (ratio >0.33):
				final_classification.append([possible_filaments[i][0],possible_filaments[i][1],0])
			else:
				final_classification.append([possible_filaments[i][0],possible_filaments[i][1],1])

		results = np.empty((len(final_classification)+1,7), dtype=object)
		for i in np.arange(len(final_classification)):
			# Table header
			results[0,:]= ['Name', 'Center Coord.', 'Box', 'Intensity', 'Area', 'Likely filament?', 'Original image']
			# Name of the filament: 1, 2, 3, 4...
			results[i+1,0]=str(i+1)

			# Coordinates of center
			results[i+1,1]= final_classification[i][0].centroid

			# Dimensions of box enclosing the region: 
			# Bounding box ``(min_row, min_col, max_row, max_col)``.
			# Pixels belonging to the bounding box are in the half-open interval
			# ``[min_row; max_row)`` and ``[min_col; max_col)``.
			results[i+1,2] = final_classification[i][0].bbox

			# Intensity of the region
			results[i+1,3] = final_classification[i][0].mean_intensity

			# Area of the region
			results[i+1,4] = final_classification[i][0].area

			# Likely a filament or not
			if final_classification[i][2]:
				results[i+1,5] = 'Probably yes'
			else:
				results[i+1,5] = 'Probably not'
			
			# From what image it comes 
			results[i+1,6] = self.path

		
		return results

	def make_image_list(initial_time, number_of_images, timestep):
		""" Make a list of the images to be fed to the time evolution algorithms. All the times have to be given
			with the format YYYYMMDDHHMMSS as a string.
			The timestep ALWAYS has to be in minutes"""
		start = datetime.strptime(initial_time, '%Y%m%d%H%M%S')
		times_list = list()
		for i in range(number_of_images):
			times_list.append(start+timedelta(minutes=i*timestep))

		times_list = [i.strftime('%Y%m%d%H%M%S') for i in times_list]
		image_names = [str(i)+'Mh.jpg' for i in times_list]
		image_list = [sun_img(i) for i in image_names]
		return image_list	


	def time_ev_filament(image_list):
		# We compute the number of images provided and run the filament detector on them
		# FILAMENT DETECTOR NO PLOT
		number_of_images = len(image_list)
		image_results = [sun_img.detecting_filaments_no_plot(i) for i in image_list]

		# Extract for the data the coordinates of the center of each region
		k = 0
		coordinates = list()
		for i in image_results:
			globals()["coord" + str(k)] = list()

			for j in i:
				if not type(j[1]) == str:
					globals()["coord" + str(k)].append(j[1])
			coordinates.append(globals()["coord" + str(k)])
			k = k+1


		# Select as our initial picture the one where most features have been detected (initial)
		# Caution: might mean we miss some if we are unlucky
		number_of_filaments = [len(i) for i in coordinates]
		initial = np.argmax(number_of_filaments)
		others = list(range(len(number_of_filaments)))
		others.remove(initial)

		# Check for spatial coincidences in the detected features among the pictures
		# First coordinate refers to the image, second coordinate to the tag of the filament in that image. 
		# Note that the one that we have chosen as a reference (initial) doesn't appear
		m = 1
		coincidences = list()

		for i in coordinates[initial]:
			globals()["same" +str(initial)+ str(m)] = list()
			for j in others:
				n = 1
				for k in coordinates[j]:
					dist = np.sqrt((k[0]-i[0])**2+(k[1]-i[1])**2)

					if dist < 10:
						globals()["same" +str(initial)+ str(m)].append((j,n))
					n = n+1
	  
			coincidences.append(globals()["same" +str(initial)+ str(m)])
			m = m+1

		# Now using the list of coincidences we obtain a list of the data of the relevant filaments in each picture

		m = 0
		iden_filaments = list()
		for i in coincidences:
			globals()["filament" + str(m)]=list()
			globals()["filament" + str(m)].append(image_results[initial][m+1])    
			for j in i:
				globals()["filament" + str(m)].append(image_results[j[0]][j[1]])
			iden_filaments.append(globals()["filament" + str(m)])
			m = m+1
		# We check how many times each structure was detected 
		number_of_coincidences = [len(i) for i in iden_filaments]

		# We get rid of the ones that are't detected half of the times
		for i in iden_filaments:
			if len(i)-1< number_of_images/2:
				iden_filaments.remove(i)


		# We count how many times our filament detecting algorithm classified the structure as "Possibly a filament"
		counts = list()
		for i in iden_filaments:
			n = 0
			for j in i:
				if j[5]== 'Probably yes':
					n = n+1
			counts.append(n)


		# Remove the structures that haven't been  classified as likely filaments more than half the times
		for i in np.arange(len(counts)):
			if counts[i]<number_of_coincidences[i]/2:
				iden_filaments.remove(iden_filaments[i])

		# Extract the area and date of each structure

		areas = list()
		images = list()
		m = 0
		for i in iden_filaments:
			globals()["areas" + str(m)] = list()
			globals()["image" + str(m)] = list()

			for j in i:
				globals()["areas" + str(m)].append(j[4])
				globals()["image" + str(m)].append(j[6])

			areas.append(globals()["areas" + str(m)])
			images.append(globals()["image" + str(m)])

			m = m+1

		# Convert the image names into dates

		dates = list()
		for i in np.arange(len(areas)):
			date =[datetime.strptime(a[:14], '%Y%m%d%H%M%S') for a in images[i]]
			dates.append(date)

		# Select as plotting image the one as a reference for the structures
		example_image = image_list[initial].original

		# Plot
		for i in np.arange(len(areas)):
			fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 6))


			ax1.imshow(example_image)
			minr, minc, maxr, maxc = iden_filaments[i][0][2]
			rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='green', linewidth=2)
			ax1.add_patch(rect)
			ax1.set_title(dates[i][initial])


			ax2.scatter(np.arange(len(areas[i])),areas[i])
			ax2.set_xticks(np.arange(0, len(areas[i]),3)) 
			ax2.grid()
			ax2.set_title('Time evolution of the filament surface')
			ax2.set_xlabel('Times')
			ax2.set_ylabel('Surface of the filament in pixels')

			ax2.set_xticklabels(dates[i][0:-1:3],rotation=45, fontsize=8)

			plt.show()



































































