from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fontforge  # noqa
import os
import multiprocessing as mp
import argparse

# conda deactivate
# apt install python3-fontforge

def convert_mp(opts):
    """Useing multiprocessing to convert all fonts to sfd files"""
    charset = opts.charset
    charset_lenw = len(str(len(opts.charset)))
    fonts_file_path = opts.ttf_path
    sfd_path = opts.sfd_path
    for root, dirs, files in os.walk(os.path.join(opts.ttf_path, opts.split)):
        ttf_fnames = files
    
    font_num = len(ttf_fnames)
    process_nums = mp.cpu_count() - 2
    font_num_per_process = font_num // process_nums + 1

    def process(process_id, font_num_p_process):
        for i in range(process_id * font_num_p_process, (process_id + 1) * font_num_p_process):
            if i >= font_num:
                break
            
            font_id = ttf_fnames[i].split('.')[0]
            split = opts.split
            font_name = ttf_fnames[i]
            
            font_file_path = os.path.join(fonts_file_path, split, font_name)
            try:
                cur_font = fontforge.open(font_file_path)
            except Exception as e:
                print('Cannot open ', font_name)
                print(e)
                continue

            target_dir = os.path.join(sfd_path, split, "{}".format(font_id))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for char_id, char in enumerate(charset):
                char_description = open(os.path.join(target_dir, '{}_{num:0{width}}.txt'.format(font_id, num=char_id, width=charset_lenw)), 'w')

                cur_font.selection.select(char)
                cur_font.copy()

                new_font_for_char = fontforge.font()
                new_font_for_char.selection.select(char)
                new_font_for_char.paste()
                new_font_for_char.fontname = "{}_".format(font_id) + font_name

                new_font_for_char.save(os.path.join(target_dir, '{}_{num:0{width}}.sfd'.format(font_id, num=char_id, width=charset_lenw)))

                char_description.write(str(ord(char)) + '\n')
                char_description.write(str(new_font_for_char[char].width) + '\n')
                char_description.write(str(new_font_for_char[char].vwidth) + '\n')
                char_description.write('{num:0{width}}'.format(num=char_id, width=charset_lenw) + '\n')
                char_description.write('{}'.format(font_id))

                char_description.close()

            cur_font.close()

    processes = [mp.Process(target=process, args=(pid, font_num_per_process)) for pid in range(process_nums)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser(description="Convert ttf fonts to sfd fonts")
    parser.add_argument("--charset", type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    parser.add_argument("--ttf_path", type=str, default='font_ttfs')
    parser.add_argument('--sfd_path', type=str, default='font_sfds')
    parser.add_argument('--split', type=str, default='train')
    opts = parser.parse_args()
    convert_mp(opts)

if __name__ == "__main__":
    main()
