"""Shared function library for the comb-content analysis.

Taken verbatim from the original ``CombContent_Analysis.ipynb``:
the "Functions" cell (mask geometry, display helpers, cell/comb
specialization, hierarchical bootstrapping) followed by the
"Bootstrapping" cell.  The bootstrapping cell redefines
``hierarchical_bootstrap`` and ``bootstrap_summary_stats`` with later
versions (the later definition wins, exactly as in the notebook) and adds
``comparative_hierarchical_bootstrap``.

Bootstrap functions use the global ``numpy.random`` stream; callers that
need reproducibility (tests, figure re-runs) seed it themselves.
"""

from __future__ import annotations

from collections import Counter
from math import floor, log10

import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad
import numpy as np
import pandas as pd
import scipy.interpolate as si
from scipy.stats import ttest_ind, ttest_rel
import seaborn as sns

from .constants import (colony_names, contenthex, simplehex, treatmenthex,  # noqa: F401
                        xmincrop, xmaxcrop, ymincrop, ymaxcrop, short_names)

# round to significant figure for statistical tests
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def jitter(xlist, amount=0.03):
    randlist = np.random.uniform(size = len(xlist), low = -amount, high = amount)
    return xlist + randlist


def centroid(mask, content):
    coords = np.argwhere(mask==content)
    centroid = np.mean(coords, axis=0)
    return centroid


def flipmasks(masks):
    # flip every other frame in a whole nest masks (takes masks for whole nest at one timepoint)
    flippedmasks = np.copy(masks)
    flippedmasks[1::2, :, :] = flippedmasks[1::2, :, ::-1] #flip every other frame
    return flippedmasks


def shiftmask(mask, hshift=3, vshift=0):
    # single mask shift
    # shift the mask inside the image canvas to align more precisely with other masks

    maskshift = np.copy(mask)

    # horizontal shift
    if hshift>0:
        clip = maskshift[:, :-hshift]
        zeroes = np.zeros((clip.shape[0], hshift))
        maskshift = np.concatenate((zeroes, clip), axis=1)

    # vertical shift
    if vshift>0:
        clip = maskshift[:-vshift, :]
        zeroes = np.zeros((vshift, clip.shape[1]))
        maskshift = np.concatenate((zeroes, clip), axis=0)

    return maskshift



def shiftmasks(masks, hshift=3, vshift=0):
    # shift every other frame in the whole nest mask shift
    # shift the masks inside the image canvas to align more precisely with other masks

    maskshift = np.copy(masks)

    # horizontal shift
    if hshift>0:
        clip = maskshift[1::2, :, :-hshift]
        zeroes = np.zeros((len(clip), clip.shape[1], hshift))
        maskshift[1::2, :, :] = np.concatenate((zeroes, clip), axis=2)

    # vertical shift
    if vshift>0:
        clip = maskshift[1::2, :-vshift, :]
        zeroes = np.zeros((len(clip), vshift, clip.shape[2]))
        maskshift[1::2, :, :] = np.concatenate((zeroes, clip), axis=1)

    return maskshift





def xyz_coords(masks, content_names_toinclude, contents=np.arange(12), zskip=4, flip=True, shift=False, colony=None, week=None):
    '''
    Get xyz coordinates of all cells of specified contents in a 3D nest mask array.
    masks: 3D numpy array of nest (z,y,x), z is frames (should be 20 for full nest)
    content_names_toinclude: list of content names to include
    colony: colony name to record for the coordinates (the notebook relied on
        a leaked global from the calling loop; callers pass colony_names[c])
    week: week index to record for the coordinates (0-based, notebook global)
    contents: list or array of content ids to extract coordinates for (default all 0-11)
    zskip: spacing between z layers (default 4, for downsampled 48)
    flip: whether to flip every other frame (default True)
    shift: whether to shift every other frame (default False)
    '''
    
    customz=[[0,0]]
    for n in range(10):
        n=customz[-1][-1] + 1
        arr = np.array([n, n+zskip])
        customz.append(arr)
    customz = np.array(customz).flatten()
    customz = customz[1:21]

    df_coords = pd.DataFrame()
    colonyf = np.copy(masks)
    if flip:
        colonyf = flipmasks(colonyf)
    if shift:
        colonyf = shiftmasks(colonyf)

    for i,content in enumerate(contents):
        dftemp = pd.DataFrame()
        coords = np.argwhere(colonyf==content)

        z = coords[:,0]
        y = coords[:,1]
        x = coords[:,2]
        zspaced = np.take(customz, z)

        dftemp['z'] = z
        dftemp['y'] = y
        dftemp['x'] = x
        dftemp['zspaced'] = zspaced
        dftemp['contentid'] = [content]*len(x)
        dftemp['content'] = [content_names_toinclude[i]]*len(x)
        dftemp['colony'] = [colony]*len(x) if colony is not None else ['']*len(x)
        dftemp['week'] = [week]*len(x) if week is not None else [0]*len(x)

        df_coords = pd.concat([df_coords,dftemp]).reset_index(drop=True)
    return df_coords


def xyz_plot(data):
    data = data[(data['content']!='background')
                 &(data['content']!='comb')
                 &(data['content']!='wood')
                 &(data['content']!='queen_cup')
                 &(data['content']!='queen_cell')
                 &(data['content']!='bees')]

    #colormap
    h = data.contentid.unique().tolist()
    hex = np.take(contenthex, h)

    x = data.x
    y = data.y
    z = data.zspaced
    contents = data.content
    import plotly.express as px  # optional viz dep (analysis extra)
    fig = px.scatter_3d(x=x, y=z, z=-y, color=contents, opacity=.15,
                   color_discrete_sequence=hex)

    fig.update_layout(
        autosize=False,
        width=1500,
        height=1000,)
    fig.show()


def arrange_nest(masks, colonyname, w, data, flip=True, shift=False):
    # Arange 3d nest "masks" into the position and orientation "w" week

    if flip:
        masks = flipmasks(masks) # flip every other mask in nest 
    if shift:
        masks = shiftmasks(masks) # shift every other mask to align 

    data = data[(data['colony']==colonyname)&
            (data['week']==w+1)&
            (data['side']=='a')
            ] 
    data = data.sort_values('position')
    frameorder = data.frame.tolist()
    inverted = data.invert_frame.tolist()
    masksoriented=[]
    for f in range(len(data)):
        if inverted[f]=='y':
            b = int((frameorder[f]-1)*2) + 1
            maskb = masks[b]
            maskb = np.flip(maskb, 1)
            masksoriented.append(maskb)

            a = b-1
            maska = masks[a]
            maska = np.flip(maska, 1)
            masksoriented.append(maska)
        else:
            a = int((frameorder[f]-1)*2)
            maska = masks[a]
            masksoriented.append(maska)

            b = a+1
            maskb = masks[b]
            masksoriented.append(maskb)

    masksoriented = np.array(masksoriented)
    
    return masksoriented


def get_class_contours(mask, class_id):
    import cv2  # optional dep (analysis extra); unused by Figures 1-5
    """ Get contours for class_id in mask. 
    
    Args:
        mask: 2D numpy array 
        class_id: value of class of interest in mask
    Return:
        list of cv2 contours
    """
    class_mask = np.where(mask==class_id, 1, 0)
    class_mask = class_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(class_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_NONE
                                          )
    
    return contours


def get_interior_mask(mask, wood_class=1):
    """Return mask of all space within the wood frame in mask of comb contents.
    
    Args:
        content_mask: 2D array of comb contents
        wood_class: value of wood in content mask
        
    Return:
        2D mask of size content mask with 1 inside frame and 0 outside.
    """
    import cv2  # optional dep (analysis extra); unused by Figures 1-5
    wood_contours = get_class_contours(mask, class_id=wood_class)
    interior_mask = np.zeros_like(mask, dtype=np.uint8)
    if len(wood_contours)>1:
        cv2.drawContours(interior_mask, wood_contours, 1, 1, -1)
    return interior_mask


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T




def showmask(mask, ax=None): # show 1 comb mask
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask, cmap='gist_stern', vmin=0, vmax=8)
    ax.axis('off')


def showmasks(masks): # show all comb masks of a colony for a week
    nrows=10
    ncols=2
    fig,ax = plt.subplots(nrows,ncols, figsize=((masks[0].shape[1]*ncols)/20,(masks[0].shape[0]*nrows)/20))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(10):
        showmask(masks[i*2], ax=ax[i,0])
        showmask(masks[i*2+1], ax=ax[i,1])



def showmask_hexagons(mask, cmap='gist_stern'):
    rows, cols = mask.shape
    fig, ax = plt.subplots(figsize=(mask.shape[1]/8,mask.shape[0]/8))

    hexagons = []
    colors = []
    dy = np.sqrt(3) / 2  # Hexagonal height spacing
    radius = 0.588  # Slightly larger than 0.5 to remove gaps

    for i in range(rows):
        for j in range(cols):
            x = j + (i % 2) * 0.5  # Offset every other row
            y = i * dy
            hexagon = patches.RegularPolygon((x, -y), numVertices=6, radius=radius, orientation=0)
            hexagons.append(hexagon)
            colors.append(mask[i, j])

    # Normalize colors using vmin and vmax
    norm = mcolors.Normalize(vmin=0, vmax=8)

    # Create a collection of hexagons
    collection = collections.PatchCollection(hexagons, array=np.array(colors), cmap=cmap, norm=norm)
    ax.add_collection(collection)
    ax.autoscale()

    plt.axis('off')  # Hide axes for aesthetics



def showmask_sheets(masks, z_spacing=200, alpha=1, elev=30, azim=140, fig=None):
    # masks = a collection of masks you want to view as sheets in 3D (use Contents[c][w][::2] for a nest of colony c at timepoint w with every other frame)
    vmin = 0
    vmax = 8
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # Normalization

    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, mask in enumerate(masks):
        z = i * z_spacing  # Assign Z position for each sheet
        x = np.arange(mask.shape[1])
        y = np.arange(mask.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Use plot_surface to create the sheet
        face_colors = plt.cm.gist_stern(norm(mask))
        ax.plot_surface(X, np.full_like(X, z), -Y, rstride=1, cstride=1, facecolors=face_colors, shade=False, alpha=alpha)

    # Adjust viewing angle
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([mask.shape[1], z_spacing, mask.shape[0]])
    ax.axis('off')





def np_area(vertices):
    # Ensure that the polygon is closed (first and last points are the same)
    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0]))

    # Apply the shoelace formula
    area = 0
    n = len(vertices)
    for i in range(n - 1):
        area += vertices[i][0] * vertices[i + 1][1]
        area -= vertices[i + 1][0] * vertices[i][1]
    area = abs(area) / 2

    return area



def sort_mode_counts(arr, ignore_zero=False):
    # Function to get the most common values and their counts in a row
    def most_common_values_and_counts(row):
        counter = Counter(row)
        if ignore_zero:
            most_common = [item for item in counter.most_common() if item[0] != 0]
        else:
            most_common = counter.most_common()
        first_value, first_count = most_common[0] if len(most_common) > 0 else (-1, -1)
        second_value, second_count = most_common[1] if len(most_common) > 1 else (-1, -1)
        return (first_value, first_count, second_value, second_count)

    # Apply the function to each row
    common_values_and_counts = np.array([most_common_values_and_counts(row) for row in arr])

    # Sort the array based on the most common value, count of the most common value,
    # second most common value, and count of the second most common value
    sorted_indices = np.lexsort((
        -common_values_and_counts[:, 3],  # Second most common value count
        -common_values_and_counts[:, 2],  # Second most common value
        -common_values_and_counts[:, 1],  # Most common value count
        -common_values_and_counts[:, 0]   # Most common value
    ))

    sorted_array = arr[sorted_indices]
    sorted_common_values_and_counts = common_values_and_counts[sorted_indices]

    # Count the total number of rows that have each most common value
    unique_first_values, first_value_counts = np.unique(sorted_common_values_and_counts[:, 0], return_counts=True)
    mode_counts = {value: count for value, count in zip(unique_first_values, first_value_counts)}

    return sorted_array, sorted_indices, mode_counts




def get_cell_sequences(masks):
    # get the sequence of content types for all cell in a given frame for a given amount of time
    # masks is a 3D array of a single comb over time (masks[week][y][x]) - you can get this by calling masks = Contents[colony][:,frame]
    #returns list of sequences for each cell and the accompanying array of their [x,y] coordinates
    slist=[]
    ptlist=[]
    for x in range(masks[0].shape[1]):
        for y in range(masks[0].shape[0]):
            pt=[x,y]
            sequence = masks[:,y,x]
            sequence = np.where(sequence>1,sequence,0)
            if np.sum(sequence)>0:
                slist.append(sequence)
                ptlist.append(pt)
    return np.array(slist), np.array(ptlist)



def get_counts_and_proportions(cell_sequences, contents_included = [0,2,5,6,7]):
    # contents_included = np.unique(cell_sequences)
    countlist=[]
    proplist=[]
    for seq in cell_sequences:
        life = np.count_nonzero(seq)
        counts=[]
        proportions=[]
        for content in contents_included:
            count = np.count_nonzero(seq==content)
            counts.append(count)
            prop = int((count/life)*100)
            proportions.append(prop)
        countlist.append(counts)
        proplist.append(proportions[1:])
    return np.array(countlist), np.array(proplist)


def crop_last2dims(arr, ymincrop=ymincrop, ymaxcrop=ymaxcrop, xmincrop=xmincrop, xmaxcrop=xmaxcrop, turn_0=True):
    """
    Crop only the last 2 dimensions of an array, regardless of how many leading dimensions.
    If turn_0 is True, set the rows before ymincrop and after ymaxcrop to 0 instead of cropping.
    """
    if turn_0:
        arr = np.copy(arr)
        # Set rows before ymincrop and after ymaxcrop to 0
        arr[..., :ymincrop, :] = 0
        arr[..., ymaxcrop:, :] = 0
        # Set columns before xmincrop and after xmaxcrop to 0
        arr[..., :, :xmincrop] = 0
        arr[..., :, xmaxcrop:] = 0
        return arr
    else:
        slicer = [slice(None)] * (arr.ndim - 2) # create a list of slice(None) for all leading dimensions
        slicer.append(slice(ymincrop, ymaxcrop)) # append the slice for the second to last dimension
        slicer.append(slice(xmincrop, xmaxcrop)) # append the slice for the last dimension
        return arr[tuple(slicer)] # use tuple to convert list to tuple for indexing


def randomize_cells_within_nests(masks, maxweek=10):
    # Randomize cells within nests, per week maintains cell content proportions but randomizes their locations within the comb area
    # masks: 4D array (week, frame, y, x)
    randomized_masks=[] # to store randomized masks for each week
    for w in range(maxweek): # iterate through weeks
        masks_week = np.copy(masks[w]) # get masks for week w
        randmasks = np.zeros(masks_week.shape) # initialize empty masks for week w
        combcoords = np.argwhere(masks_week>1) # get coordinates of comb cells

        for i in range(3,8): # iterate through content types
            ncontent = len(np.argwhere(masks_week==i)) # number of cells of content type i
            randindex = np.random.choice(len(combcoords), ncontent, replace=False) # randomly select indices from empty comb template
            randcoords = combcoords[randindex] # get coordinates of randomly selected comb cells

            randmasks[randcoords[:,0], randcoords[:,1], randcoords[:,2]] = i # fill the randomly selected comb cells with content type i

            combcoords = np.delete(combcoords, randindex, axis=0) # remove the filled comb cells from the combcoords array so they are not selected again
        
        randmasks[combcoords[:,0], combcoords[:,1], combcoords[:,2]] = 2 # fill remaining comb cells with empty comb
        randomized_masks.append(randmasks)
    return np.array(randomized_masks)




def cell_specialists(masks, maxweek=10, prop_thres=None, simplify_contents=True, print_results=True, every_other_frame=False, classify_specialists=True):

    """
    Analyze cell specialization in nest content masks.

    This function processes nest content masks (3D, 4D, or 5D numpy arrays) to classify each cell as a generalist or specialist
    (empty, brood, pollen, nectar) based on the proportion of time spent as each content type. It can also simplify content types
    and uses a threshold (mean + std by default) to define specialization.

    Parameters
    ----------
    masks : np.ndarray
        The nest content masks. Can be:
            - 3D: (week, y, x)
            - 4D: (week, frame, y, x)
            - 5D: (colony, week, frame, y, x)
    maxweek : int, optional
        The maximum number of weeks to analyze (default: 10).
    prop_thres : np.ndarray or None, optional
        Thresholds for specialization (array of 4 floats for each content type). If None, calculated as mean + std of proportions.
    simplify_contents : bool, optional
        If True, simplifies content types to: empty, brood (egg+larvae+pupae), pollen, nectar (nectar+honey).
    classify_specialists : bool, optional
        If True, classifies cells into generalist or specialist categories based on thresholds.

    Returns
    -------
    specialists : object
        An object with the following attributes:
            - sequences: np.ndarray, the content sequence for each cell
            - points: np.ndarray, the coordinates of each cell
            - counts: np.ndarray, counts of each content type per sequence
            - proportions: np.ndarray, proportions of each content type per sequence
            - highest_counts: np.ndarray, highest non-zero count per sequence
            - prop_means: np.ndarray, mean proportions (used for threshold if prop_thres is None)
            - prop_std: np.ndarray, std of proportions (used for threshold if prop_thres is None)
            - prop_thres: np.ndarray, thresholds used for classification
            - classifications: np.ndarray, assigned class for each cell (0=generalist, 2=empty, 5=brood, 6=pollen, 7=nectar)
            - classification_names: list, names of classes
            - ids: np.ndarray, unique class ids present
            - idcounts: np.ndarray, counts of each class
            - class_props: np.ndarray, percent of total cells in each class

    Prints
    ------
    - The thresholds used for each specialist class.
    - The proportion of cells in each class.

    Notes
    -----
    - The function expects the helper functions `crop_last2dims` and `get_cell_sequences` to be defined in the environment.
    - The classification id mapping is: 0=generalist, 2=empty specialist, 5=brood specialist, 6=pollen specialist, 7=nectar specialist.
    """

    # GET SEQUENCES (depending on dimensionality of "masks" input)

    if len(masks.shape) == 3: # if masks is 3D (week, y, x)
        masks = crop_last2dims(masks) # CROP only the last 2 dimensions
        s,pts = get_cell_sequences(masks) # get sequences and points
        slist = np.array(s)
        pts = np.array(pts)

    elif len(masks.shape) == 4: # if masks is 4D (week, frame, y, x)
        masks = crop_last2dims(masks) # CROP only the last 2 dimensions
        slist = []
        ptlist = []
        for f in range(masks.shape[1]):
            s,pts = get_cell_sequences(masks[:maxweek,f])
            slist.extend(s)
            ptlist.extend(pts)
        slist = np.array(slist)
        pts = np.array(ptlist)

    elif (len(masks.shape) == 1)|(len(masks.shape) == 5): # if masks is 5D (colony, week, frame, y, x)
        slist = []
        ptlist = []
        for c in range(len(masks)):
            if every_other_frame:
                masks0 = masks[c][:maxweek,::2] #all timepoints for every other comb side
            else:
                masks0 = masks[c][:maxweek] #all timepoints for every comb side
            masks0 = crop_last2dims(masks0) # CROP only the last 2 dimensions
            for f in range(masks0.shape[1]):
                s,pts = get_cell_sequences(masks0[:maxweek,f])
                slist.extend(s)
                ptlist.extend(pts)
        slist = np.array(slist)
        pts = np.array(ptlist)

    else:
        raise ValueError("Input array must be 3D, 4D, or 5D.")

    # SIMPLIFY CONTENTS (to empty, brood (egg + larvae + pupae), pollen, and nectar (honey + nectar))
    if simplify_contents:
        slist = np.where((slist>2)&(slist<6),5,slist) # Simplify brood into single value
        slist = np.where((slist==7)|(slist==8),7,slist) # Simplify nectar and honey into single value
        slist = np.where(slist>8,2,slist) # Any higher values (qcup or bees) become empty comb

    # GET COUNTS AND PROPORTIONS 
    counts, proportions = get_counts_and_proportions(slist)
    highest_counts = np.max(counts[:,1:], axis=1) # record highest non-zero count for each sequence (for visually sorting the sequences later)



    # CALCULATE SPECIALIST THRESHOLDS
    if classify_specialists:
        if prop_thres is None: # if no threshold provided, calculate it
            prop_means = np.mean(proportions, axis=0)
            prop_std = np.std(proportions, axis=0)
            prop_thres = prop_means + prop_std # threshold to be classified as specialist: need to spend proportion of time being a content type for more than the mean + 1 st dev for all DD cells

        # ASSIGN CLASSIFICATIONS TO EACH CELL
        # based on proportion of time spent as each content type, using the thresholds defined above
        classification_names = ['generalists','empty specialists','brood specialists','pollen specialists','nectar specialists']
        classifications=[]
        for seq in proportions:
            if seq[0]>prop_thres[0]:
                id = 2 # empty specialist
            elif seq[1]>prop_thres[1]:
                id = 5 # brood specialist
            elif seq[2]>prop_thres[2]:
                id = 6 # pollen specialist
            elif seq[3]>prop_thres[3]:
                id = 7 # nectar speiacialist
            else:
                id = 0 #generalist/mixed use
            classifications.append(id)
        classifications = np.array(classifications)


        # CALCULATE SPECIALIST PROPORTIONS
        ids, idcounts = np.unique(classifications, return_counts=True) # get unique ids and their counts
        class_props = (idcounts/len(slist))*100 # calculate percent of total cells in each class

        # PRINT SUMMARY
        if print_results:
            print('thresholds: ' \
                'empty specialists = ' + str(round(prop_thres[0],2)) + 
                '%; brood specialists = ' + str(round(prop_thres[1],2)) + 
                '%; pollen specialists = ' + str(round(prop_thres[2],2)) + 
                '%; nectar specialists = ' + str(round(prop_thres[3],2)) + '%')
            print('proportions: ' \
                'generalists = ' + str(round(class_props[0],2)) + 
                '%; empty specialists = ' + str(round(class_props[1],2)) + 
                '%; brood specialists = ' + str(round(class_props[2],2)) + 
                '%; pollen specialists = ' + str(round(class_props[3],2)) + 
                '%; nectar specialists = ' + str(round(class_props[4],2)) + '%')

    # OBJECT TO STORE RESULTS
    # Create a simple object to store all relevant variables
    class Specialists:
        pass
    specialists = Specialists()
    specialists.sequences = slist
    specialists.points = pts
    specialists.counts = counts
    specialists.proportions = proportions
    specialists.highest_counts = highest_counts
    if classify_specialists:
        specialists.prop_means = prop_means if prop_thres is None else np.mean(proportions, axis=0)
        specialists.prop_std = prop_std if prop_thres is None else np.std(proportions, axis=0)
        specialists.prop_thres = prop_thres
        specialists.classifications = classifications
        specialists.classification_names = classification_names
        specialists.ids = ids
        specialists.idcounts = idcounts
        specialists.class_props = class_props
    specialists.maxweek = maxweek
    specialists.masks = masks

    return specialists



def comb_specialists(masks, maxweek=10, cell_prop_thres=None, comb_prop_thres=None, reorder_frames=True, simplify_contents=True, print_results=True, every_other_frame=False, thres_calc_method = 'mean+1SD'):
    
    """
    Analyze specialization of combs across multiple colonies.

    This function determines the specialization of each comb (e.g., generalist, empty, brood, pollen, nectar) 
    by aggregating cell-level classifications over time and comparing the proportions to calculated or provided thresholds.

    Args:
        masks (list or np.ndarray): 
            A 5D array, (dimensions: [colony, week, frame, y, x]).
        maxweek (int, optional): 
            The maximum number of weeks to include in the analysis. Default is 10.
        cell_prop_thres (float or array-like, optional): 
            Threshold(s) for cell-level specialization. If None, defaults are used in cell_specialists.
        comb_prop_thres (float or array-like, optional): 
            Threshold(s) for comb-level specialization. If None, thresholds are calculated from the data.
        reorder_frames (bool, optional): 
            Whether to reorder frames so that frame 1 is always closest to the wall where the nest was initiated. Default is True.
        simplify_contents (bool, optional): 
            Whether to simplify cell contents for classification. Default is True.
        print_results (bool, optional): 
            Whether to print the calculated thresholds and summary. Default is True.

    Returns:
        CombSpecialists: 
            An object with the following attributes:
                - comb_ids (np.ndarray): 
                    Array of assigned specialist IDs for each comb and time point.
                - comb_prop_thres (np.ndarray): 
                    Array of thresholds used for comb specialization.
                - comb_class_props (list): 
                    List of class proportions for each comb and time point.

    Notes:
        - Specialist IDs correspond to: 
            0 = generalist/mixed use, 
            2 = empty specialist, 
            5 = brood specialist, 
            6 = pollen specialist, 
            7 = nectar specialist.
        - The function relies on a helper function `cell_specialists` for cell-level analysis.
    """

    categories = [0,2,5,6,7]

    # CALCULATE COMB SPECIALIZATION THRESHOLDS FIRST if not defined
    if comb_prop_thres is None:
        all_comb_class_props=[]
        for c in range(len(masks)):

            comb_class_props=[]
            for j in range(10):
                if reorder_frames: # reorder frames so that all frame 1 is closest to the wall where the nest was initiated
                    if c in [4, 5, 8]:
                        # Forward order for colonies 4, 5, 8
                        f_start = j * 2
                        f_end = f_start + 2
                    else:
                        # Reverse order for other colonies
                        f_end = 20 - j * 2
                        f_start = f_end - 2
                if every_other_frame:
                    masks0 = masks[c][:maxweek,f_start]
                else:
                    masks0 = masks[c][:maxweek,f_start:f_end] # aggregate masks for frame f side a and side b
                specialists = cell_specialists(masks0, maxweek=maxweek, prop_thres=cell_prop_thres, simplify_contents=True, print_results=False)

                # include cases where number of class type is 0
                class_props_full=[]
                for i in categories:
                    class_count = np.count_nonzero(specialists.classifications==i)
                    prop = class_count/len(specialists.sequences)
                    class_props_full.append(prop)
                comb_class_props.append(class_props_full)

            all_comb_class_props.append(comb_class_props)
        all_comb_class_props = np.array(all_comb_class_props)

        # COMB SPECIALIZATION THRESHOLDS
        comb_prop_avg = np.mean(all_comb_class_props, axis=(0,1))
        comb_prop_std = np.std(all_comb_class_props, axis=(0,1))
        if thres_calc_method == 'mean+1SD':
            comb_prop_thres = comb_prop_avg + comb_prop_std
        elif thres_calc_method == 'mean+2SD':
            comb_prop_thres = comb_prop_avg + 2*comb_prop_std
        else:
            raise ValueError("Invalid thres_calc_method. Choose 'mean+1SD' or 'mean+2SD'.")

    all_comb_ids=[]
    for c in range(len(masks)):
        comb_ids=[]
        comb_class_props=[]
        for j in range(10):
            if reorder_frames: # reorder frames so that all frame 1 is closest to the wall where the nest was initiated
                if c in [4, 5, 8]:
                    # Forward order for colonies 4, 5, 8
                    f_start = j * 2
                    f_end = f_start + 2
                else:
                    # Reverse order for other colonies
                    f_end = 20 - j * 2
                    f_start = f_end - 2

            if every_other_frame:
                masks0 = masks[c][:maxweek,f_start]
            else:
                masks0 = masks[c][:maxweek,f_start:f_end] # aggregate masks for frame f side a and side b
            specialists = cell_specialists(masks0, maxweek=maxweek, prop_thres=cell_prop_thres, simplify_contents=True, print_results=False)

            # include cases where number of class type is 0
            class_props_full=[]
            for i in categories:
                class_count = np.count_nonzero(specialists.classifications==i)
                prop = class_count/len(specialists.sequences)
                class_props_full.append(prop)
            comb_class_props.append(class_props_full)


            # ASSIGN CLASSIFICATIONS TO EACH COMB
            for comb_prop in comb_class_props:
                if class_props_full[4]>comb_prop_thres[4]:
                    id = 7 # nectar specialist
                elif class_props_full[3]>comb_prop_thres[3]:
                    id = 6 # pollen specialist
                elif class_props_full[2]>comb_prop_thres[2]:
                    id = 5 # brood specialist
                elif class_props_full[1]>comb_prop_thres[1]:
                    id = 2 # empty speiacialist
                else:
                    id = 0 #generalist/mixed use
            comb_ids.append(id)

        all_comb_ids.append(comb_ids)
    all_comb_ids = np.array(all_comb_ids)

    # PRINT RESULTS
    if print_results:
        print('comb thresholds: ' \
            'empty specialists = ' + str(round((comb_prop_thres[0]*100),2)) + 
            '%; brood specialists = ' + str(round((comb_prop_thres[1]*100),2)) + 
            '%; pollen specialists = ' + str(round((comb_prop_thres[2]*100),2)) + 
            '%; nectar specialists = ' + str(round((comb_prop_thres[3]*100),2)) + '%')
        
    # OBJECT TO STORE RESULTS
    class CombSpecialists:
        pass    
    comb_specialists = CombSpecialists()
    comb_specialists.comb_ids = all_comb_ids # 2D array of comb ids (colony, frame)
    comb_specialists.comb_prop_thres = comb_prop_thres # 1D array of comb thresholds
    comb_specialists.comb_class_props = comb_class_props # 2D array of comb class proportions (frame, class)

    return comb_specialists



def plot_cell_specialists(cell_sp, maxweek=10, figsize=(8,15), short_names=short_names, contenthex=contenthex, skip_kde=False, contents=None):

    # For empty frame to show contents in
    if contents is None:
        from .data import load_masks
        contents = load_masks()["Contents"]
    frame = contents[0][4][18]
    frame = np.where(frame==1,frame,0)

    plt.style.use('default')
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    gs = GridSpec(5, 2, figure=fig)

    # PLOT SORTED SEQUENCES
    sortidx = np.lexsort((cell_sp.highest_counts,cell_sp.classifications))
    seq_sorted = cell_sp.sequences[sortidx][::-1]

    ax1 = fig.add_subplot(gs[:,0])
    seq_sorted = np.where(seq_sorted==0,8,seq_sorted) # turn 0s to white
    ax1.imshow(seq_sorted, cmap='gist_stern', vmin=0, vmax=len(short_names)-1, interpolation='nearest', aspect='auto')
    ax1.set_xlabel('Time (weeks)')
    ax1.set_xticks(np.arange(1,cell_sp.maxweek+1,2), labels=np.arange(2,cell_sp.maxweek+1,2))
    ax1.yaxis.set_visible(False)
    ax1.spines['bottom'].set_bounds(0, 10)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('cell sequences', loc='left')


    # PLOT XY POSITIONS ON FRAME
    start=len(cell_sp.sequences)
    for i in range(len(cell_sp.ids)):
        ax = fig.add_subplot(gs[4-i,1])
        ptidx = np.argwhere(cell_sp.classifications==cell_sp.ids[i])
        points = cell_sp.points[ptidx]
        if len(cell_sp.masks.shape) == 3:
            ax.scatter(points[:,:,0],points[:,:,1], s=12, c=contenthex[cell_sp.ids[i]])
        else:
            if not skip_kde:
                sns.kdeplot(ax=ax,
                            x=points[:,0,0],
                            y=points[:,0,1],
                            fill=True,
                            alpha=.3,
                            color=contenthex[cell_sp.ids[i]],
                            )
        ax.imshow(frame, cmap='Greys', vmax=5)
        ax.set_title(cell_sp.classification_names[i], y=.9)
        ax.text(x=frame.shape[1]/2, y=frame.shape[0]+frame.shape[0]*.12, s=str(round(cell_sp.class_props[i],1)) + '%',  ha='center', va='bottom')
        ax.axis('off')

        # PLOT CLASSIFICATION LINES
        pad = len(cell_sp.sequences)*.005
        stop = start - cell_sp.idcounts[i]
        ax1.plot([maxweek-.2,maxweek-.2], [start-pad, stop+pad], c=contenthex[cell_sp.ids[i]], lw=.625*figsize[0])
        ax1.plot([maxweek-.2,maxweek+2,maxweek+3], [(start+stop)/2, (len(cell_sp.sequences)/5 )*(4-i+.5),(len(cell_sp.sequences)/5 )*(4-i+.5)], c=contenthex[cell_sp.ids[i]], lw=.375*figsize[0]) # 2 seps
        # ax1.plot([maxweek-.1,maxweek+.25, maxweek+2.25,maxweek+3], [(start+stop)/2, (start+stop)/2, (len(cell_sp.sequences)/5 )*(4-i+.5),(len(cell_sp.sequences)/5 )*(4-i+.5)], c=contenthex[cell_sp.ids[i]], lw=.375*figsize[0]) # 3 steps
        start = stop




def draw_arrow_loop(ax, radius, lw, centX, centY, angle_, theta2_, color_='black'):
    #======== Draw the arc
    arc = Arc([centX, centY], radius, radius, angle=angle_,
            theta1=0, theta2=theta2_, capstyle='round', linestyle='-', lw=lw, color=color_)
    ax.add_patch(arc)

    #======== Compute arrowhead position
    endX = centX + (radius / 2) * np.cos(rad(theta2_ + angle_))  # End position X
    endY = centY + (radius / 2) * np.sin(rad(theta2_ + angle_))  # End position Y
    orientation = rad(theta2_ + angle_ + 115)  # Tangent direction (perpendicular to radius)

    #======== Draw arrowhead (RegularPolygon)
    arrowhead = RegularPolygon(
        (endX, endY),  # Position
        numVertices=3,  # Triangle
        radius=lw / (70+(1.5*lw)),  # Scaled size
        orientation=orientation,  # Tangent alignment
        color=color_
    )
    ax.add_patch(arrowhead)

    # Set axis limits to maintain aspect ratio
    ax.set_xlim(-1.9,1.9)
    ax.set_ylim(-1.9,1.9)
    ax.set_aspect('equal')



def extrapolate_point(x1, y1, x2, y2, distance):
    # Compute direction vector
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)  # Equivalent to sqrt(dx**2 + dy**2)
    
    # Normalize to get unit direction vector
    ux, uy = dx / length, dy / length
    
    # Extrapolate the third point
    x3 = x2 + distance * ux
    y3 = y2 + distance * uy
    
    return x3, y3


def map_value(value, from_min, from_max, to_min, to_max):
    """Maps a value from one range to another."""
    return to_min + (to_max - to_min) * ((value - from_min) / (from_max - from_min))



def powspace(start, stop, power, num):
    """Generate a sequence of numbers spaced evenly on a power scale."""   
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 




def add_sig_note(ax, data, x, y, hue, order=None, hue_order=None, y_offset=0.1, method='individual'):
    """
    Add significance annotations to a seaborn boxplot.

    Parameters:
    - ax: matplotlib Axes object
    - data: pandas DataFrame
    - x, y, hue: column names for x-axis, y-axis, and hue
    - order: list of x-axis categories (optional)
    - hue_order: list of hue categories (optional)
    - y_offset: vertical offset for annotations
    - method: 'individual' t test or 'paired' t test
    """
    if order is None:
        order = data[x].unique()
    if hue_order is None:
        hue_order = data[hue].unique()

    for category in order:
        subset = data[data[x] == category]
        group1 = subset[subset[hue] == hue_order[0]][y]
        group2 = subset[subset[hue] == hue_order[1]][y]

        if len(group1) > 1 and len(group2) > 1:
            if method=='individual':
                stat, p_value = ttest_ind(group1, group2, equal_var=False)
            if method=='paired':
                stat, p_value = ttest_rel(group1, group2)
            annotation = (
                '****' if p_value <= 0.0001 else
                '***' if p_value <= 0.001 else
                '**' if p_value <= 0.01 else
                '*' if p_value <= 0.05 else
                'n.s.'
            )
            y_max = max(group1.max(), group2.max())
            ax.text(category, y_max + y_offset, annotation, ha='center', va='bottom', color='black')



def hierarchical_bootstrap(data, hierarchy_levels, method='mean', count_by=None, n_bootstraps=4000, n_jobs=-1):
    """
    Perform hierarchical bootstrapping on a pandas DataFrame.
    This function allows for bootstrapping across multiple hierarchical levels (e.g., colony, week, frame, etc.).
    For each bootstrap iteration, it samples (with replacement) the unique values at each specified hierarchy level,
    then generates all possible combinations of these sampled values. For each combination, it filters the DataFrame,
    concatenates the resulting subsets, and then samples (with replacement) from the concatenated DataFrame to compute
    the mean of all numeric columns. The process is repeated for the specified number of bootstrap iterations,
    and results are combined into a single DataFrame.
    Arguments:
      data: pandas DataFrame containing the data to bootstrap.
      hierarchy_levels: list of column names to use as hierarchical levels for bootstrapping.
      method: 'mean' to compute mean of all numeric columns, 'count' to compute mean of counts after grouping by count_by.
      count_by: column name to group by when method is 'count'.
      n_bootstraps: number of bootstrap iterations (default 4000).
      n_jobs: number of parallel jobs to run (default -1, i.e., use all processors).
    Returns:
      DataFrame of bootstrapped means for each iteration.
      """

    def single_bootstrap(): # Perform a single bootstrap iteration
        selected_hierarchy = [] # List to store selected values at each hierarchy level
        for level in hierarchy_levels: # For each hierarchy level
            unique_labels = np.unique(data[level])   # Get unique labels at this level
            selected_labels = np.random.choice(unique_labels, size=len(unique_labels), replace=True) # Bootstrap sample with replacement
            selected_hierarchy.append(selected_labels) # these are the values at each level in the hierarchy (that have themselved been selected by bootstrapping) that will be used to craft the dataset to get the final boostrapped sample from

        # Use itertools.product to get all combinations of selected values at each hierarchy level
        import builtins  # the built-in list(...) can be shadowed by notebook cells
        import itertools
        df_subsets = []

        if method == 'count':
            # Find the index of the last level in count_by within hierarchy_levels
            group_level = count_by[-1]
            group_level_idx = hierarchy_levels.index(group_level)

        # Build all combinations and their corresponding indices
        # This will create a Cartesian product of the selected values at each hierarchy level
        all_combinations = builtins.list(itertools.product(*selected_hierarchy))
        all_index_combinations = builtins.list(itertools.product(*[range(len(x)) for x in selected_hierarchy]))

        ticker = 0 # Initialize a ticker to create unique group identifiers
        # Iterate through all combinations of selected values and their indices
        for value_tuple, index_tuple in zip(all_combinations, all_index_combinations):
            df_filtered = data.copy()
            for level, value in zip(hierarchy_levels, value_tuple): # Filter the DataFrame by the selected values at each hierarchy level
                df_filtered = df_filtered[df_filtered[level] == value] 
            if method == 'count': # If method is 'count', create a group identifier based on the selected values
                group_num = ticker + index_tuple[group_level_idx] # ticker is the current group number, index_tuple[group_level_idx] is the index of the selected value at the group level
                df_filtered['group'] = group_num # Assign the group number to the filtered DataFrame
                if ((index_tuple[group_level_idx] == len(selected_hierarchy[group_level_idx]) - 1)  # Check if the index at the group level is the last one
                    & (index_tuple[-1] == len(selected_hierarchy[-1]) - 1)): # Check if the index at the last hierarchy level is also the last one
                    ticker = group_num + 1
            if not df_filtered.empty:
                df_subsets.append(df_filtered)

        # Concatenate all filtered dataframes into a single dataframe
        if len(df_subsets) == 0:
            return None
        df_result = pd.concat(df_subsets, ignore_index=True)

        # Get a bootstrapped sample
        bootstrapped_sample = df_result.sample(n=len(df_result), replace=True)

        # Calculate means or counts
        if method == 'count':   
            df_counts = bootstrapped_sample.groupby('group').sum(numeric_only=True)  # Group by count_by and sum numeric columns
            df_means = df_counts.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of counts
        if method == 'mean':
            df_means = bootstrapped_sample.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of all numeric columns

        return df_means

    from joblib import Parallel, delayed
    from tqdm.autonotebook import tqdm
    # Parallelize the bootstrapping process
    bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(single_bootstrap)() for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"))

    # Remove None results if any
    bootstrap_results = [res for res in bootstrap_results if res is not None]

    return pd.concat(bootstrap_results, ignore_index=True)




def bootstrap_summary_stats(bootstrap_samples, axis=0, lower=2.5, upper=97.5):
    """
    Compute summary statistics (median and confidence intervals) from bootstrap samples.

    Parameters:
    ----------
    bootstrap_samples : array-like
        Bootstrap samples. The shape and interpretation depend on the application.
        For example, in a histogram context, this might be (n_bootstraps x n_bins).
    axis : int, optional
        Axis along which to compute the statistics (default is 0).
    lower : float, optional
        Lower percentile for the confidence interval (default is 2.5).
    upper : float, optional
        Upper percentile for the confidence interval (default is 97.5).

    Returns:
    -------
    dict
        Contains:
        - 'median': Median values along the specified axis.
        - 'lower_bounds': Lower bounds of the confidence interval.
        - 'upper_bounds': Upper bounds of the confidence interval.
        - 'errors': Tuple of errors ([median - lower_bounds], [upper_bounds - median]).
    """
    if lower >= upper:
        raise ValueError("`lower` must be less than `upper`.")

    # Compute required percentiles
    percentiles_values = np.percentile(bootstrap_samples, [lower, 50, upper], axis=axis)
    lower_bounds, median, upper_bounds = percentiles_values

    # Calculate errors
    errors = np.array([median - lower_bounds, upper_bounds - median])
    
    return {
        'median': median,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'errors': errors
    }


def entropy(p):
    """
    Compute Shannon entropy (base 2) for a probability distribution p.
    Ignores zero probabilities to avoid log(0) errors.
    """
    p = np.array(p, dtype=float)
    p = p[p > 0]          # keep only nonzero entries
    return -np.sum(p * np.log2(p))

# Later definitions supersede the above (from the notebook's "Bootstrapping" cell).
# My functions

def hierarchical_bootstrap(data, hierarchy_levels, method='mean', count_by=None, n_bootstraps=4000, n_jobs=-1):  # noqa: F811 -- later notebook definition intentionally supersedes the above
    """
    Perform hierarchical bootstrapping on a pandas DataFrame.
    This function allows for bootstrapping across multiple hierarchical levels (e.g., colony, week, frame, etc.).
    For each bootstrap iteration, it samples (with replacement) the unique values at each specified hierarchy level,
    then generates all possible combinations of these sampled values. For each combination, it filters the DataFrame,
    concatenates the resulting subsets, and then samples (with replacement) from the concatenated DataFrame to compute
    the mean of all numeric columns. The process is repeated for the specified number of bootstrap iterations,
    and results are combined into a single DataFrame.
    Arguments:
      data: pandas DataFrame containing the data to bootstrap.
      hierarchy_levels: list of column names to use as hierarchical levels for bootstrapping.
      method: 'mean' to compute mean of all numeric columns, 'count' to compute mean of counts after grouping by count_by.
      count_by: column name to group by when method is 'count'.
      n_bootstraps: number of bootstrap iterations (default 4000).
      n_jobs: number of parallel jobs to run (default -1, i.e., use all processors).
    Returns:
      DataFrame of bootstrapped means for each iteration.
      """

    def single_bootstrap(): # Perform a single bootstrap iteration
        selected_hierarchy = [] # List to store selected values at each hierarchy level
        for level in hierarchy_levels: # For each hierarchy level
            unique_labels = np.unique(data[level])   # Get unique labels at this level
            selected_labels = np.random.choice(unique_labels, size=len(unique_labels), replace=True) # Bootstrap sample with replacement
            selected_hierarchy.append(selected_labels) # these are the values at each level in the hierarchy (that have themselved been selected by bootstrapping) that will be used to craft the dataset to get the final boostrapped sample from

        # Use itertools.product to get all combinations of selected values at each hierarchy level
        import builtins  # the built-in list(...) can be shadowed by notebook cells
        import itertools
        df_subsets = []

        if method == 'count':
            # Find the index of the last level in count_by within hierarchy_levels
            group_level = count_by[-1]
            group_level_idx = hierarchy_levels.index(group_level)

        # Build all combinations and their corresponding indices
        # This will create a Cartesian product of the selected values at each hierarchy level
        all_combinations = builtins.list(itertools.product(*selected_hierarchy))
        all_index_combinations = builtins.list(itertools.product(*[range(len(x)) for x in selected_hierarchy]))

        ticker = 0 # Initialize a ticker to create unique group identifiers
        # Iterate through all combinations of selected values and their indices
        for value_tuple, index_tuple in zip(all_combinations, all_index_combinations):
            df_filtered = data.copy()
            for level, value in zip(hierarchy_levels, value_tuple): # Filter the DataFrame by the selected values at each hierarchy level
                df_filtered = df_filtered[df_filtered[level] == value] 
            if method == 'count': # If method is 'count', create a group identifier based on the selected values
                group_num = ticker + index_tuple[group_level_idx] # ticker is the current group number, index_tuple[group_level_idx] is the index of the selected value at the group level
                df_filtered['group'] = group_num # Assign the group number to the filtered DataFrame
                if ((index_tuple[group_level_idx] == len(selected_hierarchy[group_level_idx]) - 1)  # Check if the index at the group level is the last one
                    & (index_tuple[-1] == len(selected_hierarchy[-1]) - 1)): # Check if the index at the last hierarchy level is also the last one
                    ticker = group_num + 1
            if not df_filtered.empty:
                df_subsets.append(df_filtered)

        # Concatenate all filtered dataframes into a single dataframe
        if len(df_subsets) == 0:
            return None
        df_result = pd.concat(df_subsets, ignore_index=True)

        # Get a bootstrapped sample
        bootstrapped_sample = df_result.sample(n=len(df_result), replace=True)

        # Calculate means or counts
        if method == 'count':   
            df_counts = bootstrapped_sample.groupby('group').sum(numeric_only=True)  # Group by count_by and sum numeric columns
            df_means = df_counts.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of counts
        if method == 'mean':
            df_means = bootstrapped_sample.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of all numeric columns

        return df_means

    from joblib import Parallel, delayed
    from tqdm.autonotebook import tqdm
    # Parallelize the bootstrapping process
    bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(single_bootstrap)() for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"))

    # Remove None results if any
    bootstrap_results = [res for res in bootstrap_results if res is not None]

    return pd.concat(bootstrap_results, ignore_index=True)




def comparative_hierarchical_bootstrap(data, treatment_col, treatments, hierarchy_levels, method='mean', count_by=None, n_bootstraps=4000, n_jobs=-1):
    """
    Perform hierarchical bootstrapping on a pandas DataFrame, comparing two groups. The main difference is that 
    this function calculates the difference in means (or counts) between two groups for each bootstrap iteration, 
    and the output bootstaps are the difference between the means of each sample. If the CI crosses zero, it 
    suggests that there is no significant difference between the two groups.
    
    The hierarchical approach remains the same as above.

    Arguments:
      data: pandas DataFrame containing the data to bootstrap.
      treatment_col: column name to use as the treatment variable for comparison.
      treatments: list of treatment values to compare.
      hierarchy_levels: list of column names to use as hierarchical levels for bootstrapping.
      method: 'mean' to compute mean of all numeric columns, 'count' to compute mean of counts after grouping by count_by.
      count_by: column name to group by when method is 'count'.
      n_bootstraps: number of bootstrap iterations (default 4000).
      n_jobs: number of parallel jobs to run (default -1, i.e., use all processors).
    Returns:
      DataFrame of bootstrapped differences in means (or counts) between the two groups for each iteration.
      """

    def single_bootstrap(): # Perform a single bootstrap iteration
        selected_hierarchy = [] # List to store selected values at each hierarchy level
        for level in hierarchy_levels: # For each hierarchy level
            unique_labels = np.unique(data[level])   # Get unique labels at this level
            selected_labels = np.random.choice(unique_labels, size=len(unique_labels), replace=True) # Bootstrap sample with replacement
            selected_hierarchy.append(selected_labels) # these are the values at each level in the hierarchy (that have themselved been selected by bootstrapping) that will be used to craft the dataset to get the final boostrapped sample from

        # Use itertools.product to get all combinations of selected values at each hierarchy level
        import builtins  # the built-in list(...) can be shadowed by notebook cells
        import itertools
        df_subsets = []

        if method == 'count':
            # Find the index of the last level in count_by within hierarchy_levels
            group_level = count_by[-1]
            group_level_idx = hierarchy_levels.index(group_level)

        # Build all combinations and their corresponding indices
        # This will create a Cartesian product of the selected values at each hierarchy level
        all_combinations = builtins.list(itertools.product(*selected_hierarchy))
        all_index_combinations = builtins.list(itertools.product(*[range(len(x)) for x in selected_hierarchy]))

        ticker = 0 # Initialize a ticker to create unique group identifiers
        # Iterate through all combinations of selected values and their indices
        for value_tuple, index_tuple in zip(all_combinations, all_index_combinations):
            df_filtered = data.copy()
            for level, value in zip(hierarchy_levels, value_tuple): # Filter the DataFrame by the selected values at each hierarchy level
                df_filtered = df_filtered[df_filtered[level] == value] 
            if method == 'count': # If method is 'count', create a group identifier based on the selected values
                group_num = ticker + index_tuple[group_level_idx] # ticker is the current group number, index_tuple[group_level_idx] is the index of the selected value at the group level
                df_filtered['group'] = group_num # Assign the group number to the filtered DataFrame
                if ((index_tuple[group_level_idx] == len(selected_hierarchy[group_level_idx]) - 1)  # Check if the index at the group level is the last one
                    & (index_tuple[-1] == len(selected_hierarchy[-1]) - 1)): # Check if the index at the last hierarchy level is also the last one
                    ticker = group_num + 1
            if not df_filtered.empty:
                df_subsets.append(df_filtered)

        # Concatenate all filtered dataframes into a single dataframe
        if len(df_subsets) == 0:
            return None
        df_result = pd.concat(df_subsets, ignore_index=True)

        # Get a bootstrapped sample
        bootstrapped_sample = df_result.sample(n=len(df_result), replace=True)

        # Calculate means or counts and then compute the difference between the two treatment groups
        if method == 'count':   
            df_counts_a = bootstrapped_sample[bootstrapped_sample[treatment_col] == treatments[0]].groupby('group').sum(numeric_only=True)  # Group by count_by and sum numeric columns for treatment A
            df_counts_b = bootstrapped_sample[bootstrapped_sample[treatment_col] == treatments[1]].groupby('group').sum(numeric_only=True)  # Group by count_by and sum numeric columns for treatment B
            df_means_a = df_counts_a.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of counts for treatment A
            df_means_b = df_counts_b.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of counts for treatment B
            df_means = df_means_a.subtract(df_means_b)  # Difference between means of treatment A and B
        if method == 'mean':
            df_a = bootstrapped_sample[bootstrapped_sample[treatment_col] == treatments[0]]
            df_b = bootstrapped_sample[bootstrapped_sample[treatment_col] == treatments[1]]
            df_means_a = df_a.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of all numeric columns for treatment A
            df_means_b = df_b.mean(numeric_only=True).to_frame().T.rename(columns=lambda x: str(x)) # Mean of all numeric columns for treatment B
            df_means = df_means_a.subtract(df_means_b)  # Difference between means of treatment A and B

        return df_means

    from joblib import Parallel, delayed
    from tqdm.autonotebook import tqdm
    # Parallelize the bootstrapping process
    bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(single_bootstrap)() for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"))

    # Remove None results if any
    bootstrap_results = [res for res in bootstrap_results if res is not None]

    return pd.concat(bootstrap_results, ignore_index=True)




def bootstrap_summary_stats(bootstrap_samples, axis=0, lower=2.5, upper=97.5):  # noqa: F811 -- later notebook definition intentionally supersedes the above
    """
    Compute summary statistics (median and confidence intervals) from bootstrap samples.

    Parameters:
    ----------
    bootstrap_samples : array-like
        Bootstrap samples. The shape and interpretation depend on the application.
        For example, in a histogram context, this might be (n_bootstraps x n_bins).
    axis : int, optional
        Axis along which to compute the statistics (default is 0).
    lower : float, optional
        Lower percentile for the confidence interval (default is 2.5).
    upper : float, optional
        Upper percentile for the confidence interval (default is 97.5).

    Returns:
    -------
    dict
        Contains:
        - 'median': Median values along the specified axis.
        - 'lower_bounds': Lower bounds of the confidence interval.
        - 'upper_bounds': Upper bounds of the confidence interval.
        - 'errors': Tuple of errors ([median - lower_bounds], [upper_bounds - median]).
    """
    if lower >= upper:
        raise ValueError("`lower` must be less than `upper`.")

    # Compute required percentiles
    percentiles_values = np.percentile(bootstrap_samples, [lower, 50, upper], axis=axis)
    lower_bounds, median, upper_bounds = percentiles_values

    # Calculate errors
    errors = np.array([median - lower_bounds, upper_bounds - median])
    
    return {
        'median': median,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'errors': errors
    }
