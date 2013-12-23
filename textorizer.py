#!/usr/bin/python

import Image
import ImageFont
import ImageDraw
import argparse
import numpy as np
from scipy import ndimage
from sys import argv,path,version
from _version import __version__


def load_image(input_image):
    print ('loading image')
    return Image.open(input_image)


def filter_image(img, image_filter):
    print ('filtering')
    print (img.size)
    # Convert to BW
    img_grey = img.convert('L')
    imgnp = np.asarray(img_grey)

    # Gaussian + Median Filters
    imgnp = ndimage.gaussian_filter(imgnp, 2)
    imgnp = [np.left_shift(np.right_shift(x, 6), 7) for x in imgnp]
    imgnp = ndimage.median_filter(imgnp, 3)

    if cfg["DEF_INVERT"] is True:
        imgnp = np.invert(imgnp)

    print ((imgnp[0]))
    return imgnp


def apply_text(input_image, input_text, output_image, font=None,
                font_size=None, font_color=None, font_margin=None,
                 variable_size=False):
    print ('loading text')
    with open(input_text) as f:
        data = f.read()

    im_w = len(input_image[0])
    im_h = len(input_image)
    print ((im_w, im_h))

    if font_size is None:
        font_size = cfg["DEF_FONT_SIZE"] + 1
    if font is None:
        font = ImageFont.truetype(cfg["DEF_FONT"], font_size)
    else:
        font = ImageFont.truetype(font, font_size)
    if font_color is None:
        font_color = cfg["DEF_FONT_COLOR"]
    if font_margin is None:
        font_margin = cfg["DEF_MARGIN"]

    output = Image.fromarray(np.uint8(input_image))
    draw = ImageDraw.Draw(output)
    data_to_draw = np.zeros((im_w/font_size, im_h/font_size))
    print (data)
    # Convert array to 1D
    #img1d = np.reshape(output,(im_w*im_h))

    # Static font size
    for i in range(0, im_h / font_size):
        draw.text((0, i*font_size), data[i*im_w:i*im_w + im_w], font_color, font = font)

    #i = 0
    #for data in img1d:
        #for line in range(0, im_h / font_size):
            ##for c in range(i*im_w, i*im_w + im_w):
                ##print (input_image[line][c:c+font_size])
                ###if (np.sum(input_image[c:c+font_size])/font_size) > 128:
                    ###data_to_draw.append(data[i])
                ###else:
                    ###data_to_draw.append(' ')
            #i = i + 1
    #for i in range(0, im_h / font_size):
        #draw.text((0, i*font_size), data_to_draw[i*im_w:i*im_w + im_w], font_color, font = font)

    # Non static font size

    if output_image is None:
        output_image = 'tmp.png'
    output.save(output_image, "PNG")


def main(argv):
    args = argv[1].split()
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--version',action='version',version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('-i', dest='i', nargs='?', required =True, help="input image")
    parser.add_argument('-o', dest='o', nargs='?', help='output image name (optional)')
    parser.add_argument('-t', dest='t', nargs='?', required =True, help='input text (txt only)')
    parser.add_argument('-f', dest='f', nargs='?', help='filter applied (optional)')
    args = parser.parse_args(args)

    input_image = args.i
    image_filter = args.f
    output_image = args.o
    input_text = args.t

    img = load_image(input_image)
    img = filter_image(img, image_filter)
    img = apply_text(img, input_text, output_image)


if __name__ == "__main__":
    cfg = {}
    if version < 3:
        execfile("config.txt", cfg)
    else:
        exec(open("config.txt").read(), cfg)
    main(argv)
