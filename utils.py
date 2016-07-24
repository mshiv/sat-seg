import numpy as np


def color_image(image, num_classes=2):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

#def image_writer(image, num_classes=2):
#	import