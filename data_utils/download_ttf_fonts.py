import os
import urllib.request

glyphazzn_url_lines = open('svg_vae_data/glyphazzn_urls.txt', 'r').readlines()
retrieved_glyphazzn_url_lines = open('svg_vae_data/glyphazzn_urls_downloaded.txt', 'w')

root_font_dir = 'svg_vae_data/ttf_fonts'
if not os.path.exists(root_font_dir):
    os.makedirs(os.path.join(root_font_dir, 'train'))
    os.makedirs(os.path.join(root_font_dir, 'test'))

retrieved_count = 0

for line in glyphazzn_url_lines:
    font_id, split, url = line.strip().split(', ')
    font_name = url.split('/')[-1]
    font_dir = os.path.join(root_font_dir, split)
    try:
        urllib.request.urlretrieve(url, os.path.join(font_dir, font_name))
    except Exception as e:
        print(e)
        continue
    retrieved_count += 1
    retrieved_glyphazzn_url_lines.write(line)
    if retrieved_count % 10 == 0:
        print(retrieved_count)

print("Retrieved", retrieved_count)

retrieved_glyphazzn_url_lines.close()
