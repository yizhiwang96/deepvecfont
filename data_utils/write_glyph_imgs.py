from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import argparse
import numpy as np
import os

def write_glyph_imgs(opts):

    for root, dirs, files in os.walk(os.path.join(opts.ttf_path, opts.split)):
        ttf_names = files
    ttf_names.sort()
    for fontid in range(len(ttf_names)):
        fontname = ttf_names[fontid].split('.')[0]
        print(fontname)
        g_idx_dict = {}
        ttf_file_path = os.path.join(opts.ttf_path, opts.split, ttf_names[fontid])

        try:
            font = ImageFont.truetype(ttf_file_path, opts.img_size, encoding="unic")
        except:
            print('cant open ' + fontname)
            continue

        fontimgs_array = np.zeros((len(opts.alphabet), opts.img_size, opts.img_size), np.uint8)
        fontimgs_array[:, :, :] = 255

        for charid in range(len(opts.alphabet)):
            # read the meta file
            txt_fpath = os.path.join(opts.sfd_path, opts.split, fontname, fontname + '_' + "%02d"%(charid) + '.txt')
            try:
                txt_lines = open(txt_fpath,'r').read().split('\n')
            except:
                print('cannot read text file')
                continue

            # the offsets are calculated according to the rules in data_utils/svg_utils.py
            vbox_w = float(txt_lines[1])
            vbox_h = float(txt_lines[2])
            norm = max(int(vbox_w), int(vbox_h))

            if int(vbox_h) > int(vbox_w):
                add_to_y = 0
                add_to_x = abs(int(vbox_h) - int(vbox_w)) / 2
                add_to_x = add_to_x * (64.0 / norm)
            else:
                add_to_y = abs(int(vbox_h) - int(vbox_w)) / 2
                add_to_y = add_to_y * (64.0 / norm)
                add_to_x = 0

            char = opts.alphabet[charid]
            array = np.ndarray((opts.img_size,opts.img_size), np.uint8)
            array[:, :] = 255
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)

            try:
                font_width, font_height = font.getsize(char)
            except:
                print('cant calculate height and width ' + "%04d"%fontid + '_' + "%02d"%(charid))
                continue
            
            try:
                ascent, descent = font.getmetrics()
            except:
                print('cannot get ascent, descent')
                continue

            delta = (opts.img_size - (descent + ascent)) /2
            draw.text((add_to_x, add_to_y + opts.img_size-ascent-int((opts.img_size/24.0)*(4.0/3.0))), char, (0) ,font=font)
            fontimgs_array[charid] = np.array(image)
            
        np.save(os.path.join(opts.sfd_path, opts.split, fontname, 'imgs_' + str(opts.img_size) + '.npy'), fontimgs_array)

def main():
    parser = argparse.ArgumentParser(description="Write glyph images")
    parser.add_argument("--alphabet", type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    parser.add_argument("--ttf_path", type=str, default='font_ttfs')
    parser.add_argument('--sfd_path', type=str, default='font_sfds')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--split', type=str, default='train')
    opts = parser.parse_args()

    write_glyph_imgs(opts)


if __name__ == "__main__":
    main()

