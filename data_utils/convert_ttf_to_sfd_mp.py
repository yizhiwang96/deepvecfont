from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fontforge  # noqa
import os
import multiprocessing as mp
import argparse


# need python2, apt install python-fontforge

def get_font_id_split_name(opts):
    """Verfiying every downloaded fonts, and get font_id, split, font_name"""
    alphabet_chars = opts.alphabet
    valid_fonts_urls = open(opts.downloaded_fonts_urls_file, 'r').readlines()
    font_id_split_name = open(opts.font_id_split_name_file, 'w')
    fonts_file_path = opts.ttf_path

    font_id = 0
    for font_line in valid_fonts_urls:
        font_name = font_line.strip().split(', ')[-1].split('/')[-1]
        split = font_line.strip().split(', ')[1]

        font_file_path = os.path.join(fonts_file_path, split, font_name)
        try:
            # open a font
            cur_font = fontforge.open(font_file_path)
        except Exception as e:
            print('Cannot open', font_name)
            print(e)
            continue

        try:
            # select all the 52 chars
            for char in alphabet_chars:
                cur_font.selection.select(char)
        except Exception as e:
            print(font_name, 'does not have all chars')
            print(e)
            continue

        font_id_split_name.write("{:06d}".format(font_id) + ', ' + split + ', ' + font_name + '\n')

        font_id += 1

    font_id_split_name.close()


def convert_mp(opts):
    """Useing multiprocessing to convert all fonts to sfd files"""
    alphabet_chars = opts.alphabet
    valid_fonts = open(opts.font_id_split_name_file, 'r').readlines()

    fonts_file_path = opts.ttf_path
    sfd_path = opts.sfd_path

    process_nums = mp.cpu_count() - 2
    lines_num = len(valid_fonts)
    lines_num_per_process = lines_num // process_nums

    def process(process_id, line_num_p_process):
        for i in range(process_id * line_num_p_process, (process_id + 1) * line_num_p_process):
            if i >= lines_num:
                break
            font_line = valid_fonts[i]
            font_id = font_line.strip().split(', ')[0]
            split = font_line.strip().split(', ')[1]
            font_name = font_line.strip().split(', ')[-1]

            font_file_path = os.path.join(fonts_file_path, split, font_name)
            try:
                cur_font = fontforge.open(font_file_path)
            except Exception as e:
                print('Cannot open', font_name)
                print(e)
                continue

            target_dir = os.path.join(sfd_path, split, "{}".format(font_id))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for char_id, char in enumerate(alphabet_chars):
                char_description = open(os.path.join(target_dir, '{}_{:02d}.txt'.format(font_id, char_id)), 'w')

                cur_font.selection.select(char)
                cur_font.copy()

                new_font_for_char = fontforge.font()
                new_font_for_char.selection.select(char)
                new_font_for_char.paste()
                new_font_for_char.fontname = "{}_".format(font_id) + font_name

                new_font_for_char.save(os.path.join(target_dir, '{}_{:02d}.sfd'.format(font_id, char_id)))

                char_description.write(str(ord(char)) + '\n')
                char_description.write(str(new_font_for_char[char].width) + '\n')
                char_description.write(str(new_font_for_char[char].vwidth) + '\n')
                char_description.write('{:02d}'.format(char_id) + '\n')
                char_description.write('{}'.format(font_id))

                char_description.close()

            cur_font.close()

    processes = [mp.Process(target=process, args=(pid, lines_num_per_process)) for pid in range(process_nums + 1)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser(description="Convert ttf fonts to sfd fonts")
    parser.add_argument("--alphabet", type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    parser.add_argument("--downloaded_fonts_urls_file", type=str, default='svg_vae_data/glyphazzn_urls_downloaded.txt')
    parser.add_argument("--font_id_split_name_file", type=str, default='svg_vae_data/font_id_split_name.txt')
    parser.add_argument("--ttf_path", type=str, default='svg_vae_data/ttf_fonts')
    parser.add_argument('--sfd_path', type=str, default='svg_vae_data/sfd_font_glyphs_mp')

    opts = parser.parse_args()

    get_font_id_split_name(opts)
    convert_mp(opts)


if __name__ == "__main__":
    main()
