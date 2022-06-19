#!/usr/bin/python
from matplotlib import gridspec, cm, dates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

## COLORS
# N colors distributed evenly across a color map
# see https://matplotlib.org/examples/color/colormaps_reference.html for color maps
def make_N_colors(cmap_name, N):
     cmap = cm.get_cmap(cmap_name, N)
     cmap = cmap(np.arange(N))[:,0:3]
     cmap = np.fliplr(cmap)
     return [tuple(i) for i in cmap]

## BOXPLOTS from a list of vectors
def plotData(matrix,plotTitle,yLabelText,plotLabels,statTest = 'k'):

	palette = 'spring'
	boxData = matrix
	figWidth = int(3.5 * len(boxData))
	if len(boxData) > 2:
		numPlots = len(boxData)
		plotColors = make_N_colors(palette,numPlots)
		plotColors = plotColors * numPlots
		plotLabels = plotLabels * numPlots
		figWidth = figWidth * numPlots * 0.5
	else:
		plotColors = ['k','r'] # make_N_colors(palette,numPlots)

	f = plt.figure(num=None, figsize=(figWidth, 10), dpi=80, facecolor='w', edgecolor='k')
	ax = f.add_subplot(111)

	b1 = ax.boxplot(boxData, widths=0.5, sym = '')
	b1 = formatBoxColors(b1,plotColors)

	# do some scatter plots to add the data
	xPos = 1
	for b in boxData:
		numPoints = len(b)
		xPoints=wobbleAround(xPos,numPoints,0.1)
		# add the points!
		plt.scatter(xPoints,b,c=[plotColors[xPos-1]],alpha=0.2,s=100)
		xPos += 1

	ax.set_xticklabels(plotLabels, fontsize = 18)
	ax.yaxis.label.set_size(18)
	#ax.set_ylim([-1.1,1.1])

	ax.set_ylabel(yLabelText,fontsize=18)

	if len(boxData) == 2:
		pval = statsFromBoxData(boxData,statTest)[0]
		print(pval)
		# titleLabel = plotTitle + ('; p = %1.3f' % pval)
		titleLabel = plotTitle
	else:
		statsFromBoxData(boxData,statTest)
		titleLabel = plotTitle

	# comment ON to show title with p-value
	ax.set_title(titleLabel,fontsize=18)

	#f.set_tight_layout(True)
	#ax.set_ylim([0,205])
	ax.tick_params(axis='y',labelsize=16)
	plt.show()
	return f, ax

# format colors of a boxplot object
def formatBoxColors(bp, plotColors):
	boxColors = plotColors
	baseWidth = 3
	for n,box in enumerate(bp['boxes']):
		box.set( color=boxColors[n], linewidth=baseWidth)

	for n,med in enumerate(bp['medians']):
		med.set( color=boxColors[n], linewidth=baseWidth)

	bdupes=[]
	for i in boxColors:
		bdupes.extend([i,i])

	boxColors = bdupes
	for n,whisk in enumerate(bp['whiskers']):
		#whisk.set( color=(0.1,0.1,0.1), linewidth=2, alpha = 0.5)
		whisk.set( color=boxColors[n], linewidth=baseWidth, alpha = 0.5)

	for n,cap in enumerate(bp['caps']):
		cap.set( color=boxColors[n], linewidth=baseWidth, alpha = 0.5)

	return bp

# stats from boxplot data
def statsFromBoxData(boxData,statTest):
	pvals = []

	# check if we should logit transform the data
	needTransform = False

	# for b in boxData:
	# 	if np.min(b) >= 0 and np.max(b) <= 1:
	# 		needTransform = True
	# 		#pass
	if needTransform == True:
		print('transformed the data!')
		boxData = logitBoxData(boxData)
		#print(boxData)

	for i in range(len(boxData)):
		for j in range(i+1,len(boxData)):
			if statTest in ['k','kruskal','kruskalwallis','kw']:
				_,p = stats.kruskal(boxData[i],boxData[j])
				print('%i vs. %i: %1.3f by Kruskal-Wallis' % (i+1,j+1,p))
				pvals.append(p)
			if statTest in ['t','tt','ttest']:
				_,p = stats.ttest_ind(boxData[i],boxData[j])
				print('%i vs. %i: %1.3f by ttest-ind' % (i+1,j+1,p))
				pvals.append(p)
			# MORE STAT TESTS?
	print('')

	return pvals

# points to scatter on boxplot
def wobbleAround(center,number,distAway):
	# to find points to add to box/whisker plots, centered around midline of box
	import random
	l=[]
	while len(l) < number:
		l.append(random.uniform(center-distAway, center+distAway))
	return l

# logit transform
def logitVec(d):
	# convert to proportions if not done already
	if np.max(d)>1:
		d = d / float(max(d))

	# logit is log ( y / [1 - y] )
	# add some to take care of 1's and 0's
	e = np.min(d[np.where(d>0)])
	num = d + e
	dem = (1-d) + e

	return np.log(num/dem)

def logitBoxData(boxData):
	transformed = []
	for b in boxData:
		transformed.append(logitVec(b))
	return transformed
