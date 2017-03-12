



import sys
import getopt
import csv
import os
import re
import numpy as np
import random
import math
import time
from sklearn.cluster import MiniBatchKMeans

sys.path.append('/usr/local/apps/opencv/opencv-2.4.11/lib/python2.7/site-packages/')

import cv2


class Rect():
	def __init__(self, new_size, image, orig_line=None):
		#size is determined by (height, width)		
		self.size = new_size
		self.image = image
		#loc is (x, y)
		self.loc = (0, 0)
		self.reshaped_image = None
		self.orig_line = orig_line

	def overlap(self, other):
		left = max(self.loc[0], other.loc[0])
		right = min(self.loc[0] + self.size[1], other.loc[0] + other.size[1])

		bottom = max(self.loc[1], other.loc[1])
		top = min(self.loc[1] + self.size[0], other.loc[1] + other.size[0])

		if (left > right or bottom > top):
			return 0
		else:
			return (right - left) * (top - bottom)

	def shared_side_length(self, other):
		length = 0
		if (self.loc[0] + self.size[1] == other.loc[0] or self.loc[0] == other.loc[0] + other.size[1]):
			bottom = max(self.loc[1], other.loc[1])
			top = min(self.loc[1] + self.size[0], other.loc[1] + other.size[0])

			if (bottom < top):
				length += top - bottom

		if (self.loc[1] + self.size[0] == other.loc[1] or self.loc[1] == other.loc[1] + other.size[0]):
			left = max(self.loc[0], other.loc[0])
			right = min(self.loc[0] + self.size[1], other.loc[0] + other.size[1])
			if (left < right):
				length += right - left

		return length

	def csv_line(self):
		new_line = self.orig_line + [self.loc[0], self.loc[1], self.size[0], self.size[1]]
		return new_line

	def placed_load(self, loc, size):
		self.loc = loc
		self.size = size



def read_met_csv(csv_file):

	non_decimal = re.compile(r'[^\d.]+')
	rect_list = []

	met_csv = csv.reader(open(csv_file, 'rb'), delimiter=',')

	for row in met_csv:
		image_filename = row[-1]

		image = cv2.imread(image_filename)

		size_str = row[25]

		begin = size_str.find('(')

		size_str = size_str[begin + 1:]

		splits = size_str.split()

		print(splits[0] + " " + splits[2] + "\n" + size_str + " " + image_filename)
		
		height = float(non_decimal.sub('', splits[0])) * 10
		
		splits[2] = splits[2].replace('cm', ' ')
		width = float(non_decimal.sub('', splits[2])) * 10

		new_rect = Rect((height, width), image, row)

		rect_list.append(new_rect)

	return rect_list

def read_placed_list(csv_file):
	non_decimal = re.compile(r'[^\d.]+')
	rect_list = []

	met_csv = csv.reader(open(csv_file, 'rb'), delimiter=',')

	for row in met_csv:
		image_filename = row[-5]

		image = cv2.imread(image_filename)

		size_str = row[25]

		begin = size_str.find('(')

		size_str = size_str[begin + 1:]

		splits = size_str.split()

		print(splits[0] + " " + splits[2] + "\n" + size_str + " " + image_filename)
		
		height = float(non_decimal.sub('', splits[0])) * 10
		
		splits[2] = splits[2].replace('cm', ' ')
		width = float(non_decimal.sub('', splits[2])) * 10

		new_rect = Rect((height, width), image, row[:-4])

		new_rect.placed_load((int(float(row[-4])), int(float(row[-3]))), (int(float(row[-2])), int(float(row[-1]))))

		rect_list.append(new_rect)

	return rect_list

def get_match_pts(rect):

	intervals = [0, 0.5, 1]

	matched_pts = []

	for x_interval in intervals:

		x_shift = int(x_interval * rect.size[1])

		for y_interval in intervals:
			y_shift = int(y_interval * rect.size[0])

			x_loc = rect.loc[0] + x_shift
			y_loc = rect.loc[1] + y_shift
			matched_pts.append((x_loc, y_loc))

	return matched_pts

def get_aligned_locs(placement, new_rect):
	intervals = [0, 0.5, 1]

	aligned_locs = []

	for x_interval in intervals:

		x_shift = int(x_interval * new_rect.size[1])

		for y_interval in intervals:
			y_shift = int(y_interval * new_rect.size[0])

			x_loc = placement[0] - x_shift
			y_loc = placement[1] - y_shift

			aligned_locs.append((x_loc, y_loc))

	return aligned_locs

def look_ud(placed, rect_in, x_loc, ud):

	rect = placed[rect_in]
	
	y_loc = rect.loc[1] if ud == 0 else rect.loc[1] + rect.size[0]

	min_distance = -1
	pot_loc = None	
	for i in range(len(placed)):
		if (i == rect_in):
			continue

		if (placed[i].loc[0] > x_loc or placed[i].loc[0] + placed[i].size[1] < x_loc):
			continue

		placed_y_loc = placed[i].loc[1] if ud == 1 else placed[i].loc[1] + placed[i].size[0]

		diff = y_loc - placed_y_loc if ud == 0 else placed_y_loc - y_loc

		if (diff <= 0):
			continue

		if (min_distance == -1 or diff < min_distance):
			min_distance = diff

			pot_loc = (int(x_loc), int(placed_y_loc))

	return min_distance, pot_loc


def look_lr(placed, rect_in, y_loc, lr):
	
	rect = placed[rect_in]
	
	x_loc = rect.loc[0] if lr == 0 else rect.loc[0] + rect.size[1]

	min_distance = -1
	pot_loc = None

	for i in range(len(placed)):
		if (i == rect_in):
			continue

		if (placed[i].loc[1] > y_loc or placed[i].loc[1] + placed[i].size[0] < y_loc):
			continue

		placed_x_loc = placed[i].loc[0] if lr == 1 else placed[i].loc[0] + placed[i].size[1]

		diff = x_loc - placed_x_loc if lr == 0 else placed_x_loc - x_loc

		if (diff <= 0):
			continue

		if (min_distance == -1 or diff < min_distance):
			min_distance = diff
			pot_loc = (int(placed_x_loc), int(y_loc))

	return min_distance, pot_loc
			

def get_valid_locs(placed, new_rect):

	valid_locs = []

	for rect in placed:
		placed_match_pts = get_match_pts(rect)

		for pos_placement in placed_match_pts:
			aligned_locs = get_aligned_locs(pos_placement, new_rect)

			for pos_loc in aligned_locs:
				new_rect.loc = pos_loc

				overlap_sum = 0

				for existing_rect in placed:
					overlap_sum += new_rect.overlap(existing_rect)

				
				if (overlap_sum == 0):
					valid_locs.append(pos_loc)

	for i, rect in enumerate(placed):

		interval = [0, 1]
		for x_int in interval:
			for y_int in interval:
				min_distance, pot_loc = look_ud(placed, i, rect.loc[0] + x_int * rect.size[1], y_int)
	
				aligned_locs_ud = []
				if (min_distance > 0):
					aligned_locs_ud = get_aligned_locs(pot_loc, new_rect)
				
				min_distance, pot_loc = look_lr(placed, i, rect.loc[1] + y_int * rect.size[0], x_int)

				aligned_locs_lr = []
				if (min_distance > 0):
					aligned_locs_lr = get_aligned_locs(pot_loc, new_rect)

				aligned_locs = aligned_locs_ud + aligned_locs_lr

				#print(aligned_locs)
				for pos_loc in aligned_locs:
					new_rect.loc = pos_loc

					overlap_sum = 0

					for existing_rect in placed:
						overlap_sum += new_rect.overlap(existing_rect)

				
					if (overlap_sum == 0):
						valid_locs.append(pos_loc)

				
		

	return valid_locs

def pt_distance(a, b):

	x_size = max(a[0], b[0]) - min(a[0], b[0])
	y_size = max(a[1], b[1]) - min(a[1], b[1])

	return math.pow((x_size * x_size) + (y_size * y_size), 0.5)


def max_rect_distance(a, b):
	distances = []

	interval = [0, 1]

	for ax_int in interval:
		for ay_int in interval:
			a_pt = (a.loc[0] + ax_int * a.size[1], a.loc[1] + ay_int * a.size[0])
			for bx_int in interval:
				for by_int in interval:
					b_pt = (b.loc[0] + bx_int * b.size[1], b.loc[1] + by_int * b.size[0])
					distances.append(pt_distance(a_pt, b_pt))

	return max(distances)
			

def corner_distance(rect_list):

	max_distance = 0

	intervals = [0, 1]
	for a in rect_list:
		for b in rect_list:
			rect_dis = max_rect_distance(a, b)

			if (rect_dis > max_distance):
				max_distance = rect_dis

	return max_distance

def new_rect_max_distance(rect_list, new_rect):

	max_distance = 0
	for a in rect_list:
		rect_dis = max_rect_distance(a, new_rect)

		if (rect_dis > max_distance):
			max_distance = rect_dis

	return max_distance

def max_corner_distances(placed, new_rect, valid_locs):
	
	distances = []

	cur_max_dis = corner_distance(placed)

	for temp_loc in valid_locs:
		new_rect.loc = temp_loc

		new_rect_dis = new_rect_max_distance(placed, new_rect)
		
		distances.append(max(new_rect_dis, cur_max_dis))

	#print(distances)
	return distances



def adjust_rects(placed, out_im_size):

	min_x = 0
	min_y = 0
	max_x = 0
	max_y = 0

	for rect in placed:
		if (rect.loc[0] < min_x):
			min_x = rect.loc[0]
		if (rect.loc[1] < min_y):
			min_y = rect.loc[1]

		if (rect.loc[0] + rect.size[1] > max_x):
			max_x = rect.loc[0] + rect.size[1]
		if (rect.loc[1] + rect.size[0] > max_y):
			max_y = rect.loc[1] + rect.size[0]


	if (out_im_size != -1):
		min_x = ((max_x - min_x) / 2) + min_x - (out_im_size / 2)
		min_y = ((max_y - min_y) / 2) + min_y - (out_im_size / 2)

	for rect in placed:
		temp_loc = rect.loc
		
		rect.loc = (int(temp_loc[0] - min_x), int(temp_loc[1] - min_y))

			
	if (out_im_size == -1):
		return (int(max_y) + 1 - min_y, int(max_x) + 1 - min_x)
	else:
		return (out_im_size, out_im_size)

	
def test_create_image(placed, out_name):

	for rect in placed:
		print("Rect")
		print(rect.loc)
		print(rect.size)
	max_size = adjust_rects(placed)

	print(max_size)

	test_im = np.zeros((max_size[0], max_size[1], 3), np.uint8)

	for rect in placed:
		r = random.randint(0, 255)
		g = random.randint(0, 255)
		b = random.randint(0, 255)

		for x in range(rect.loc[0], rect.loc[0] + int(rect.size[1])):
			for y in range(rect.loc[1], rect.loc[1] + int(rect.size[0])):
				test_im[y, x, 0] = r
				test_im[y, x, 1] = g
				test_im[y, x, 2] = b

	print(test_im.shape)
	print(out_name)
	cv2.imwrite(out_name, test_im)


def local_avg(im, placed, avg_size=100):

	im_shape = im.shape
	shift = avg_size
	
	avg_im = np.zeros((im_shape[0] + 2 * shift, im_shape[1] + 2 * shift, im_shape[2]), np.uint8)

	for y in range(im_shape[0]):
		print("Percent complete %02f" % ((y / float(im_shape[0]))* 100))
		for x in range(im_shape[1]):
			x_low = x - shift / 2
			x_high = x + shift / 2
			y_low = y - shift / 2
			y_high = y + shift / 2

			if (x_low < 0):
				x_low = 0
			if (y_low < 0):
				y_low = 0
			if (y_high >= im_shape[0]):
				y_high = im_shape[0] - 1
			if (x_high >= im_shape[1]):
				x_high = im_shape[1] - 1
				
			for channel in range(im_shape[2]):
				
				acc = np.sum(im[y_low:y_high,x_low:x_high,channel], dtype="float64")
				avg = acc / (shift * shift)

				avg_im[y + shift, x + shift, channel] = avg.astype("uint8")

	for rect in placed:
		print("Adjusted")
		print(rect.loc)
		print(rect.size)
		print(rect.image.shape)
		reshape_size = (int(rect.size[1]), int(rect.size[0]))
		reshaped_im = cv2.resize(rect.image, reshape_size)

		print(reshaped_im.shape)
		

		avg_im[rect.loc[1] + shift:rect.loc[1] + int(rect.size[0]) + shift,rect.loc[0] + shift:rect.loc[0] + int(rect.size[1]) + shift, :] = reshaped_im[:,:,:]

	return avg_im


def calc_dis(a, b):
	y_dif = a[0] - b[0]
	x_dif = a[1] - b[1]

	return math.sqrt((x_dif * x_dif) + (y_dif * y_dif))

def x_bounds(placed, rect_in):
	l_bound = 0
	r_bound = 0

	for i in range(len(placed)):
		if (i == rect_in):
			continue

		y_beg = max(placed[rect_in].loc[1], placed[i].loc[1])
		y_end = min(placed[rect_in].loc[1] + placed[rect_in].size[0], placed[i].loc[1] + placed[i].size[0])

		if (y_end <= y_beg):
			continue

		x_dis = placed[rect_in].loc[0] - placed[i].loc[0]

		if (x_dis > 0):
			l_bound = 1

		if (x_dis < 0):
			r_bound = 1

	return l_bound + r_bound

def y_bounds(placed, rect_in):
	u_bound = 0
	d_bound = 0

	for i in range(len(placed)):
		if (i == rect_in):
			continue

		x_beg = max(placed[rect_in].loc[0], placed[i].loc[0])
		x_end = min(placed[rect_in].loc[0] + placed[rect_in].size[1], placed[i].loc[0] + placed[i].size[1])

		if (x_end <= x_beg):
			continue

		y_dis = placed[rect_in].loc[1] - placed[i].loc[1]

		if (y_dis > 0):
			u_bound = 1

		if (y_dis < 0):
			d_bound = 1

	return u_bound + d_bound
		

def get_edge_rects(placed):
	edges = []

	for rect_in in range(len(placed)):
		
		x_bound = x_bounds(placed, rect_in)
		y_bound = y_bounds(placed, rect_in)

		if (x_bound < 2 or y_bound < 2):
			edges.append(1)
		else:
			edges.append(0)

	return edges

def kmeans_dominant_color(placed):

	total_size = 0
	for rect in placed:
		total_size += rect.size[0] * rect.size[1]

	all_pixels = np.zeros((int(total_size), 3), dtype="float64")

	start_index = 0
	for rect in placed:
		size = int(rect.size[0] * rect.size[1])
		all_pixels[start_index:start_index + size,:] = rect.reshaped_im.reshape((size, 3)).astype("float64") / 255.0
		start_index += size
		
	np.random.shuffle(all_pixels)

	kmeans = MiniBatchKMeans(n_clusters=5, max_iter=500)
	kmeans.fit(all_pixels)

	counts = np.bincount(kmeans.labels_)

	for i in range(len(kmeans.cluster_centers_)):
		print counts[i], kmeans.cluster_centers_[i,:]

	dom_cluster =  kmeans.cluster_centers_[np.argmax(counts), :]

	dom_cluster = dom_cluster * 255

	return dom_cluster.astype('uint8')

def rect_dom_colors(placed):

	kmeans = MiniBatchKMeans(n_clusters=5, max_iter=500)
	for rect in placed:
		pixels = rect.reshaped_im.reshape((int(rect.size[0] * rect.size[1]), 3)).astype("float64") / 255.0

		np.random.shuffle(pixels)
	
		kmeans.fit(pixels)

		counts = np.bincount(kmeans.labels_)

		dom_cluster =  kmeans.cluster_centers_[np.argmax(counts), :]

		dom_cluster = dom_cluster * 255

		print np.amax(counts) / float(rect.size[0] * rect.size[1]), dom_cluster

		rect.dom_color = dom_cluster.astype('uint8')

			


def get_space_around(rects, rect_in):
	u_space = 8000
	d_space = 8000
	l_space = 8000
	r_space = 8000

	for i in range(len(rects)):
		
		if (i == rect_in):
			continue

		x_beg = max(rects[rect_in].loc[0], rects[i].loc[0])
		x_end = min(rects[rect_in].loc[0] + rects[rect_in].size[1], rects[i].loc[0] + rects[i].size[1])
	
		y_beg = max(rects[rect_in].loc[1], rects[i].loc[1])
		y_end = min(rects[rect_in].loc[1] + rects[rect_in].size[0], rects[i].loc[1] + rects[i].size[0])

		if (x_end > x_beg):
			y_dis = rects[rect_in].loc[1] - rects[i].loc[1]

			if (y_dis > 0):
				space = rects[rect_in].loc[1] - (rects[i].loc[1] + rects[i].size[0]) 
				if (space < u_space):
					u_space = space

			if (y_dis < 0):
				space = rects[i].loc[1] - (rects[rect_in].loc[1] + rects[rect_in].size[0])
				if (space < d_space):
					d_space = space
		
		if (y_end > y_beg):
			x_dis = rects[rect_in].loc[0] - rects[i].loc[0]

			if (x_dis > 0):
				space = rects[rect_in].loc[0] - (rects[i].loc[0] + rects[i].size[1])
				if (space < l_space):
					l_space = space

			if (x_dis < 0):
				space = rects[i].loc[0] - (rects[rect_in].loc[0] + rects[rect_in].size[1])
				if (space < r_space):
					r_space = space

	if (u_space == 8000 or d_space == 8000 or l_space == 8000 or r_space == 8000):
		print "Really weird", u_space, d_space, l_space, r_space
		u_space = 0
		d_space = 0
		l_space = 0
		r_space = 0
 
	return l_space, r_space, u_space, d_space

	
def center_unbounded_rects(rects):

	edges = get_edge_rects(rects)

	pixels_shifted = 500
	while (pixels_shifted > 0):
		pixels_shifted = 0
		for i, rect in enumerate(rects):
			if (edges[i]):
				continue

			left_space, right_space, up_space, down_space = get_space_around(rects, i)
			x_shift = int((left_space - right_space) / 2)
			y_shift = int((up_space - down_space) / 2)
			
			rect.loc = (rect.loc[0] - x_shift, rect.loc[1] - y_shift)

			pixels_shifted += abs(x_shift) + abs(y_shift)

		print("Pixels shifted %d" % (pixels_shifted))
		

def global_image_avg(placed, im):
	im_shape = im.shape
	shift = rad_len
	
	avg = np.zeros((3,), np.uint8)
	image_pixels = 0.0	
	for rect in placed:
		image_pixels += rect.size[0] * rect.size[1]

	for channel in range(im_shape[2]):
		acc = np.sum(im[:,:,channel], dtype="float64")

		channel_avg = acc / image_pixels

		avg[channel] = channel_avg.astype(np.uint8)

	#print(avg)
	return avg

def edge_radii(placed, edges, mid_pt):
	min_rad = 10000000
	max_rad = 0
	for i, rect in enumerate(placed):
		if (edges[i] == 0):
			continue

		interval = [0, 1]
		for x_int in interval:
			for y_int in interval:
				cur_pt = (rect.loc[1] + (y_int * rect.size[0]), rect.loc[0] + (rect.size[1] * x_int))

				dis = calc_dis(mid_pt, cur_pt)

				if (dis < min_rad):
					min_rad = dis
				if (dis > max_rad):
					max_rad = dis
	return min_rad, max_rad
	
		
def avg_background(im, placed, rad_len=100, fade="none", min_scale_factor=0):
	im_shape = im.shape
	shift = rad_len
	
	avg = kmeans_dominant_color(placed)
	print(avg)
	
	avg_im = np.zeros((im_shape[0] + 2 * shift, im_shape[1] + 2 * shift, im_shape[2]), np.uint8)
	

	edges = get_edge_rects(placed)

	print "edge number", sum(edges)
	mid_pt = (float(im.shape[0]) / 2, float(im.shape[1]) / 2)

	print "mid pt", mid_pt

	min_rad, max_rad = edge_radii(placed, edges, mid_pt)	

	
	print "Rads: ", min_rad, max_rad
	for y in range(avg_im.shape[0]):
		for x in range(avg_im.shape[1]):
			'''			
			dis = calc_dis(mid_pt, (y, x))

			if (dis > max_rad):
				continue

			if (dis <= min_rad):
				avg_im[y, x,:] = avg[:]
				continue
			'''
			for channel in range(3):
				scale_factor = 1
				if (fade == "max_rad"):
					pass
				elif (fade == "image_edge"):
					dis = calc_dis(mid_pt, (y, x))
					max_dis = calc_dis(mid_pt, (0, 0))
					scale = min_scale_factor + (1.0 - min_scale_factor) * (1 - ((max_dis - dis) / (max_dis)))
					
					
				grad_avg = avg[channel].astype(np.float64) * scale_factor
				avg_im[y, x, channel] = grad_avg.astype(np.uint8)


	
	for i, rect in enumerate(placed):
		print("Adjusted")
		print(rect.loc)
		print(rect.size)
		print(rect.image.shape)
		reshaped_im = rect.reshaped_im
		if (reshaped_im is None):
			reshape_size = (int(rect.size[1]), int(rect.size[0]))
			reshaped_im = cv2.resize(rect.image, reshape_size)

		print(reshaped_im.shape)
		

		avg_im[rect.loc[1] + shift:rect.loc[1] + int(rect.size[0]) + shift,rect.loc[0] + shift:rect.loc[0] + int(rect.size[1]) + shift, :] = reshaped_im[:,:,:]
 
	
	return avg_im

def get_rect_mask(avg_im, placed, buf):

	mask = np.zeros((avg_im.shape[0], avg_im.shape[1]))

	for rect in placed:
		mask[rect.loc[1] + buf:rect.loc[1] + int(rect.size[0]) + buf,rect.loc[0] + buf:rect.loc[0] + int(rect.size[1]) + buf] = 1

	return mask

def pt_to_rect(rect, pt):

	x_dis = None

	if (rect.loc[0] > pt[0]):
		x_dis = rect.loc[0] - pt[0]
	
	elif (rect.loc[0] + rect.size[1] < pt[0]):
		x_dis = pt[0] - (rect.loc[0] + rect.size[1])

	elif (rect.loc[0] <= pt[0]  and rect.loc[0] + rect.size[1] >= pt[0]):
		x_dis = 0
	y_dis = None
		
	if (rect.loc[1] > pt[1]):
		y_dis = rect.loc[1] - pt[1]
	
	elif (rect.loc[1] + rect.size[0] < pt[1]):
		y_dis = pt[1] - (rect.loc[1] + rect.size[0])

	elif (rect.loc[1] <= pt[1] and rect.loc[1] + rect.size[0] >= pt[1]):
		y_dis = 0

	return math.sqrt((x_dis * x_dis) + (y_dis * y_dis))


def mask_blur(im, rect_mask, filter_size):

	mask = np.ones(shape=rect_mask.shape) - rect_mask

	acc_im = np.zeros(shape=im.shape, dtype="float64")
	mask_acc = np.zeros(shape=mask.shape, dtype="float64")
	f_im = im.astype("float64")

	start_time = time.time()
	for y in range(im.shape[0]):
		for x in range(im.shape[1]):

			if (y == 0 and x == 0):
				acc_im[y, x, :] = f_im[y, x, :]
				mask_acc[y, x] = mask[y, x]
				continue
			
			if (y == 0):
				acc_im[y, x, :] = f_im[y, x, :] + acc_im[y, x - 1, :]
				mask_acc[y, x] = mask[y, x] + mask_acc[y, x - 1]				
				continue			
			if (x == 0):
				acc_im[y, x, :] = f_im[y, x, :] + acc_im[y - 1, x, :]
				mask_acc[y, x] = mask[y, x] + mask_acc[y - 1, x]				
				continue
			
			acc_im[y, x, :] = f_im[y, x, :] + acc_im[y - 1, x, :] + acc_im[y, x - 1, :] - acc_im[y - 1, x - 1, :]
			mask_acc[y, x] = mask[y, x] + mask_acc[y - 1, x] + mask_acc[y, x - 1] - mask_acc[y - 1, x- 1]

		if (y % 10 == 0):
			time_so_far = time.time() - start_time
			eta = (time_so_far * (im.shape[0] + im.shape[0])) / (y + 1)
			sys.stdout.write("Blurred %d / %d completed, eta: %.2f\r" % (y + 1, im.shape[0] + im.shape[0], eta))
			sys.stdout.flush()
			
	blurred_im = np.zeros(shape=im.shape, dtype="uint8")
	for y in range(im.shape[0]):
		for x in range(im.shape[1]):
	#for y in range(100):
	#	for x in range(100):
			if (rect_mask[y, x] == 1):
				continue
	
			x_low = max(x - filter_size, 0)
			y_low = max(y - filter_size, 0)

			x_high = min(x + filter_size, im.shape[1] - 1)
			y_high = min(y + filter_size, im.shape[0] - 1)

			total_avg = acc_im[y_high, x_high, :]
			mask_num = mask_acc[y_high, x_high]

			if (x_low > 0):
				total_avg = total_avg[:] - acc_im[y_high, x_low - 1, :]
				mask_num -= mask_acc[y_high, x_low - 1]			
			if (y_low > 0):
				total_avg = total_avg[:] - acc_im[y_low - 1, x_high, :]
				mask_num -= mask_acc[y_low - 1, x_high]
			if (x_low > 0 and y_low > 0):
				total_avg = total_avg[:] + acc_im[y_low - 1, x_low - 1, :]
				mask_num += mask_acc[y_low - 1, x_low - 1]
			
			if (mask_num == 0):
				continue

			total_avg = total_avg / mask_num

			blurred_im[y, x, :] = total_avg[:].astype("uint8")
		
		if (y % 10 == 0):
			time_so_far = time.time() - start_time
			eta = (time_so_far * (im.shape[0] * im.shape[0])) / (y + im.shape[0] + 1)
			sys.stdout.write("Blurred %d / %d completed, eta: %.2f\r" % (y + im.shape[0] + 1, im.shape[0] + im.shape[0], eta))
			sys.stdout.flush()

	return blurred_im

def scalar_darken(im, scalar):

	f_im = im.astype("float32")

	f_im = f_im * scalar

	return f_im.astype("uint8")

def get_enclosed_rects(placed, mask):
	search_rects = []
	interval = [0, 1]
	for rect in placed:
		searchable = False
		
		for x in range(int(rect.loc[0]), int(rect.loc[0] + rect.size[1])):
			
			if (mask[rect.loc[1] - 1, x] == 0 or mask[rect.loc[1] + rect.size[0] + 1, x] == 0):
				searchable = True

		for y in range(int(rect.loc[1]), int(rect.loc[1] + rect.size[0])):
			if (mask[y, rect.loc[0] - 1] == 0 or mask[y, rect.loc[1] + 1] == 0):
				searchable = True

		if searchable:
			search_rects.append(rect)

	print("Removed %d rects from search list!" % (len(placed) - len(search_rects)))

	return search_rects


def distance_fade(im, mask, fade_dis, search_rects):
	fade_im = im.astype("float32")

	start_time = time.time()

	for y in range(im.shape[0]):
		for x in range(im.shape[1]):
	#for y in range(300):
	#	for x in range(300):
			if (mask[y, x] == 1):
				continue
			
			distances = []
			for i, rect in enumerate(search_rects):
				dis = pt_to_rect(rect, (x, y))
				distances.append((dis, i))

			#print(distances)
			distances.sort()
			#print(distances)
			

			intensity = 0.0

			if (distances[0][0] < fade_dis):
				intensity = 1 - (distances[0][0] / fade_dis)
				#print(intensity)

			fade_im[y, x, :] = fade_im[y, x, :] * intensity
		
		if (y % 10 == 0):
			time_so_far = time.time() - start_time
			eta = (time_so_far * im.shape[0]) / (y + 1)
			sys.stdout.write("Distance Fade %d / %d completed, Total time: %.2f, Remaining Time: %.2f\r" % (y + 1, im.shape[0], eta / 60, ((1.0 - (y / im.shape[0]) * eta) / 60.0)))
			sys.stdout.flush()

	return fade_im.astype("uint8")
	
			

def local_color_background(im, placed, out_dir, buf=0, fade="top two"):
	
	im_shape = im.shape
	shift = buf
	
	rect_dom_colors(placed)

	avg_im = np.zeros((im_shape[0] + 2 * shift, im_shape[1] + 2 * shift, im_shape[2]), np.uint8)
	

	mask = get_rect_mask(avg_im, placed, buf)
	sharp_back = os.path.join(out_dir, "local_background.png")
	start_time = time.time()
	search_rects = get_enclosed_rects(placed, mask)
	
	if os.path.exists(sharp_back):
		avg_im = cv2.imread(sharp_back)
	else:
		
		eta = 500.0
		for y in range(avg_im.shape[0]):
			for x in range(avg_im.shape[1]):
				if (mask[y, x] == 1):
					continue
			
				distances = []
				for i, rect in enumerate(search_rects):
					dis = pt_to_rect(rect, (x - buf, y - buf))
					distances.append((dis, i))

				#print(distances)
				distances.sort()
				#print(distances)

				#return
				if (fade == "none"):
					avg_im[y, x, :] = search_rects[distances[0][1]].dom_color
				elif (fade == "top two"):
					total_dis = 0.0				
					for i in range(2):
						total_dis += distances[i][0]

					if (total_dis == 0):
						avg_im[y, x, :] = search_rects[distances[0][1]].dom_color
						continue

				
					grad_avg = np.zeros((1, 3), dtype="float64")
				
					for i in range(2):
						share = 1.0 - (distances[i][0] / total_dis)

						#share = share / 2.0
				
						grad_avg = grad_avg + (search_rects[distances[i][1]].dom_color.astype("float64") * share)
				
					avg_im[y, x,:] = grad_avg[:].astype("uint8")
			if (y % 10 == 0):
				time_so_far = time.time() - start_time
				eta = (time_so_far * avg_im.shape[0]) / (y + 1)
				sys.stdout.write("Grad avg %d / %d completed, Total time: %.2f, Remaining Time: %.2f\r" % (y + 1, avg_im.shape[0], eta / 60, ((1.0 - (y / avg_im.shape[0]) * eta) / 60.0)))
				sys.stdout.flush()
	
	cv2.imwrite(sharp_back, avg_im)

	print("Total time taken: %.2f" % ((time.time() - start_time) / 60))
	
	print("Blurring")
	avg_im = mask_blur(avg_im, mask, 300)

	#avg_im = scalar_darken(avg_im, 0.8)

	avg_im = distance_fade(avg_im, mask, 2400, search_rects)

	#f_avg_im = avg_im.astype("float32") * 0.75
	#avg_im = f_avg_im.astype("uint8")		
	
	print("Done blurring")
	for i, rect in enumerate(placed):
		print("Adjusted")
		print(rect.loc)
		print(rect.size)
		print(rect.image.shape)
		reshaped_im = rect.reshaped_im
		if (reshaped_im is None):
			reshape_size = (int(rect.size[1]), int(rect.size[0]))
			reshaped_im = cv2.resize(rect.image, reshape_size)

		print(reshaped_im.shape)
		

		avg_im[rect.loc[1] + buf:rect.loc[1] + int(rect.size[0]) + buf,rect.loc[0] + buf:rect.loc[0] + int(rect.size[1]) + buf, :] = reshaped_im[:,:,:]
 
			
	return avg_im
		
			


def create_image(placed, out_name, out_im_size, out_dir, add_background=False):
	
	for rect in placed:
		print("Rect")
		print(rect.loc)
		print(rect.size)
	max_size = adjust_rects(placed, out_im_size)	


	
	center_unbounded_rects(placed)

	
	print(max_size)

	test_im = np.zeros((max_size[0], max_size[1], 3), np.uint8)


	for rect in placed:
		print("Adjusted")
		print(rect.loc)
		print(rect.size)
		print(rect.image.shape)
		reshape_size = (int(rect.size[1]), int(rect.size[0]))
		reshaped_im = cv2.resize(rect.image, reshape_size)
		rect.reshaped_im = reshaped_im

		print(reshaped_im.shape)
		

		for x in range(rect.loc[0], rect.loc[0] + int(rect.size[1])):
			for y in range(rect.loc[1], rect.loc[1] + int(rect.size[0])):
				test_im[y, x, :] = reshaped_im[y - rect.loc[1],x - rect.loc[0],:]
				

		#print(test_im[ rect.loc[1]:rect.loc[1] + rect.size[0], rect.loc[0]:rect.loc[0] + rect.size[1],:].shape)
		#test_im[ rect.loc[0]:rect.loc[0] + rect.size[1], rect.loc[1]:rect.loc[1] + rect.size[0],:] = reshaped_im[:,:,:]

	print(test_im.shape)
	print(out_name)

	#kmeans_dominant_color(placed)

	if (add_background):
		test_im = local_color_background(test_im, placed, out_dir)

	#test_im = avg_background(test_im, placed)

	cv2.imwrite(out_name, test_im)

def shared_length(rect_list, new_rect):
	length = 0
	for rect in rect_list:
		length += new_rect.shared_side_length(rect)

	return length

def get_shared_side_lengths(placed, new_rect, locs):
	
	shared_lengths = []

	for loc in locs:
		new_rect.loc = loc

		shared_lengths.append(shared_length(placed, new_rect))

	return shared_lengths

def place_rectangles(rect_list):
	placed = [rect_list[0]]

	valid_locs_time = 0
	max_distance_time = 0
	cur_rect = 1

	iter_time = time.time()

	for new_rect in rect_list[1:]:
		#finding all places where we can place the new rectangle
		temp_time = time.time()		
		valid_locs = get_valid_locs(placed, new_rect)
		valid_locs_time += time.time() - temp_time

		#valid_locs.append((-2000, -660))
		#print(valid_locs)
		temp_time = time.time()
		max_distances = max_corner_distances(placed, new_rect, valid_locs)
		max_distance_time += time.time() - temp_time
		loc_index = max_distances.index(min(max_distances))

		#print(max_distances)

		min_dis_locations = []
		min_dis = min(max_distances)
		for i in range(len(max_distances)):
			if (max_distances[i] == min_dis):
				min_dis_locations.append(valid_locs[i])

		shared_side_lengths = get_shared_side_lengths(placed, new_rect, min_dis_locations)

		loc_index = shared_side_lengths.index(max(shared_side_lengths))			
				
		new_rect.loc = min_dis_locations[loc_index]

		placed.append(new_rect)
		print("Placed rect: %d in %f seconds" % (cur_rect, time.time() - iter_time))
		cur_rect += 1
		iter_time = time.time()

	print("Valid locs time: %f" % (valid_locs_time))
	print("max distance time: %f" % (max_distance_time))

	return placed

def remove_duplicates(first, second):

	no_dups = []

	for rect in second:
		dup = False
		for r in first:
			if (rect.orig_line[0] == r.orig_line[0]):
				dup = True

		if not dup:
			no_dups.append(rect)

	print("Removed: %d duplicated" % (len(second) - len(no_dups)))
	return no_dups

def place_secondary_rectangles(placed, second):
	valid_locs_time = 0
	max_distance_time = 0
	cur_rect = len(placed)

	iter_time = time.time()

	distance_threshold = corner_distance(placed)

	for new_rect in second:	
		temp_time = time.time()		
		valid_locs = get_valid_locs(placed, new_rect)
		valid_locs_time += time.time() - temp_time

		#valid_locs.append((-2000, -660))
		#print(valid_locs)
		temp_time = time.time()
		max_distances = max_corner_distances(placed, new_rect, valid_locs)
		max_distance_time += time.time() - temp_time

		if (min(max_distances) <= distance_threshold):
			loc_index = max_distances.index(min(max_distances))
			min_dis_locations = []
			min_dis = min(max_distances)
			for i in range(len(max_distances)):
				if (max_distances[i] == min_dis):
					min_dis_locations.append(valid_locs[i])

			shared_side_lengths = get_shared_side_lengths(placed, new_rect, min_dis_locations)

			loc_index = shared_side_lengths.index(max(shared_side_lengths))			
				
			new_rect.loc = min_dis_locations[loc_index]

			placed.append(new_rect)
			print("Placed rect: %d in %f seconds" % (cur_rect, time.time() - iter_time))
		else:
			print("Tried rect: %d in %f seconds" % (cur_rect, time.time() - iter_time))
		
		cur_rect += 1
		iter_time = time.time()

	print("Valid locs time: %f" % (valid_locs_time))
	print("max distance time: %f" % (max_distance_time))

	return placed



def output_placement(placed, out_name):

	with open(out_name, 'wb') as csv_file:
		im_writer = csv.writer(csv_file, delimiter=',')

		for rect in placed:
			print(rect.csv_line())
			im_writer.writerow(rect.csv_line())

			

def main(argv):
	
	opts, args = getopt.getopt(argv, "i:o:s:p:b", ["csv=", "out=", "size=", "prefix=", "second_list=", "overwrite", "background",
							"placement_csv="])

	csv_filename = ""
	out_dir = ""
	out_im_size = -1
	prefix = ""
	second_csv = ""
	overwrite_csv = False
	add_background = False
	placement_csv = ""

	for opt, arg in opts:
		if opt in ("--csv", "-i"):
			csv_filename = arg
		elif opt in ("--out", "-o"):
			out_dir = arg
		elif opt in ("--size", "-s"):
			out_im_size = int(arg)
		elif opt in ("--prefix", "-p"):
			prefix = arg
		elif opt in ("--second_list"):
			second_csv = arg
		elif opt in ("--overwrite"):
			overwrite_csv = True
		elif opt in ("--background", "-b"):
			add_background = True
		elif opt in ("--placement_csv"):
			placement_csv = arg

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	out_name = os.path.join(out_dir, prefix + '_collage.png')

	rect_out_name = os.path.join(out_dir, prefix + "_placement.csv")

	if (placement_csv != ""):
		rect_out_name = placement_csv

	placed = None
	if (os.path.exists(rect_out_name) and not overwrite_csv):
		placed = read_placed_list(rect_out_name)
	else:
		rect_list = read_met_csv(csv_filename)

		rect_list.sort(key=lambda x: x.size[0] * x.size[1], reverse=True)

		placed = place_rectangles(rect_list)

		if (second_csv != ""):
			second_list = read_met_csv(second_csv)
			second_list = remove_duplicates(rect_list, second_list)
			second_list.sort(key=lambda x: x.size[0] * x.size[1], reverse=True)
			place_secondary_rectangles(placed, second_list)

	
	output_placement(placed, rect_out_name)
	
	#test_create_image(placed, "other" + out_name)

	create_image(placed, out_name, out_im_size, out_dir, add_background=add_background)

	
	

		
	
	


if __name__ == "__main__":
	main(sys.argv[1:])
