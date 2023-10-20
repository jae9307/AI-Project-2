"""
painter.py
Creates art using a genetic algorithm
Authors: Jose Estevez, Brett Lubberts
"""

import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse


def recombine(
	im1: np.ndarray, im2: np.ndarray
) -> np.ndarray:
	"""Create a new image from two images. 

	Vars:
		im1: the first image
		im2: the second image

	Returns:
		A new image, chosen by first, randomly choosing
		between the horizontal or vertical orientation,
		and then slicing each image into two pieces along
		a randomly-chosen vertical or horizontal line.
	"""
	recombination = random.randint(0, 1)
	if recombination == 0:
		# vertical line split
		split = random.randint(0, im1.shape[1])
		left = im1[:, :split, :]
		right = im2[:, split:, :]

		new = np.hstack((left, right))

		return new
	else:
		# horizontal line split
		split = random.randint(0, im1.shape[0])
		top = im1[:split, :, :]
		bottom = im2[split:, :, :]

		new = np.vstack((top, bottom))

		return new


def mutate(im: np.ndarray) -> np.ndarray:
	"""Mutate an image.

	Vars:
		im: the image to mutate.

	Returns:
		A new image, which is the same as the original,
		except that on of the colors is the image is
		globally (i.e., everywhere it occurs in the image)
		replace with a randomly chosen new color.
	"""

	current_color = im[random.randint(0, im.shape[0]-1),
					random.randint(0, im.shape[1]-1), :]

	new_color = np.array([random.randint(0, 255),
						  random.randint(0, 255), random.randint(0, 255)])

	im[np.where((im == current_color).all(axis=2))] = new_color
	return im


def evaluate(im: np.ndarray):
	"""Evaluate an image.

	Vars:
		im: the image to evaluate.

	Returns:
		The value of the evaluation function on im.
		Since art is subjective, you have complete
		freedom to implement this however you like.
	"""
	# just sums the red and greens and subtracts the blues
	#val = im[:,:,0] + im[:,:, 1] - im[:, :, 2]
	#return np.sum(val)
	different_neighbors = 0
	for (x,y) in zip(range(400), range(800)):
		if x + 1 < 400 and y < 800:
			if im[x][y].all() != im[x+1][y].all():
				different_neighbors +=1
		if x - 1 >= 0 and y < 800:
			if im[x][y].all() != im[x-1][y].all():
				different_neighbors +=1
		if x < 400 and y - 1 >= 0:
			if im[x][y-1].all() != im[x][y-1].all():
				different_neighbors +=1
		if x < 400 and y + 1 < 800:
			if im[x][y+1].all() != im[x][y+1].all():
				different_neighbors +=1

	return different_neighbors


def main():
	parser = argparse.ArgumentParser(
    	prog='painter',
    	description='creates paintings according to a genetic algorithm'
	)

	parser.add_argument('-g', '--generations',
			default=200, help="The number of generations to run", type=int)
	parser.add_argument('-p',
				'--pools', default=15, help="The size of the pool", type=int)
	parser.add_argument('-m', '--mutation', default=.6,
						help="The chance of a mutation", type=float)
	parser.add_argument('-r', '--recombine', default = 2,
		help="The number of pairs to recombine in each generation", type=int)
	args = parser.parse_args()

	red = np.zeros((400,800,3))
	red[:,:,0] = 255

	blue = np.zeros((400,800,3))
	blue[:,:,2] = 255

	pool = [red, blue]

	for i in range(args.generations):
		new_pool = []
		for j in range(args.pools):
			im1 = random.choice(pool)
			im2 = random.choice(pool)

			new_im = recombine(im1, im2)

			if random.random() < args.mutation:
				new_im = mutate(new_im)
			new_pool.append(new_im)

		eval_scores = []
		for j in range(len(new_pool)):
			eval_scores.append(evaluate(new_pool[j]))

		combined = list(zip(eval_scores, new_pool))

		combined.sort(key=lambda x: x[0], reverse=True)

		sorted_evals, sorted_pool = zip(*combined)

		pool = sorted_pool[:args.pools]

	for i in range(3):
		name = "art" + str(i+1) + ".tiff"
		plt.imsave(name, pool[i]/255)
	# uncomment the lines below to view the image
	#plt.imshow(blue)
	#plt.show() 

	
if __name__ == '__main__':
	main()

