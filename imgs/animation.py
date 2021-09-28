import re
import os
import numpy as np
import bezier

def cal_bezier_length(degree, coord):
    nodes = np.asfortranarray(coord)
    curve = bezier.Curve(nodes, degree=degree)
    length = curve.length
    return length

def parse_svg_rel(svg_c):
    path_ = svg_c.split('d="')[1]
    path = path_.split('" fill=')[0]
    path_splited = re.split(r"([mlc])", path)
    new_paths = []
    new_paths_lengths = []
    cur_x = 0.0
    cur_y = 0.0
    first_move = True
    for idx in range(0,len(path_splited)):
        if len(path_splited[idx]) == 0: continue
        if path_splited[idx] == 'm':
            coords_str = path_splited[idx+1]
            if first_move:
                x4 = float(coords_str.split(' ')[1])
                y4 = float(coords_str.split(' ')[2])
                first_move = False
            else:
                x4 = cur_x + float(coords_str.split(' ')[1])
                y4 = cur_y + float(coords_str.split(' ')[2])        
            cur_x = x4
            cur_y = y4

        if path_splited[idx] == 'l':
            cur_path = 'M ' + str(cur_x) + ' ' + str(cur_y) + ' '
            coords_str = path_splited[idx+1]
            x4 = cur_x + float(coords_str.split(' ')[1])
            y4 = cur_y + float(coords_str.split(' ')[2])
            new_paths_lengths.append(cal_bezier_length(1, [[float(cur_x), float(x4)],[float(cur_y), float(y4)]]))
            cur_path += 'L ' + str(x4) + ' ' + str(y4) + ' '
            cur_x = x4
            cur_y = y4
            new_paths.append(cur_path)

        if path_splited[idx] == 'c':
            cur_path = 'M ' + str(cur_x) + ' ' + str(cur_y) + ' '
            coords_str = path_splited[idx+1]
            x1 = cur_x
            y1 = cur_y
            x2 = cur_x + float(coords_str.split(' ')[1])
            y2 = cur_y + float(coords_str.split(' ')[2])
            x3 = cur_x + float(coords_str.split(' ')[3])
            y3 = cur_y + float(coords_str.split(' ')[4])
            x4 = cur_x + float(coords_str.split(' ')[5])
            y4 = cur_y + float(coords_str.split(' ')[6])
            new_paths_lengths.append(cal_bezier_length(3, [[float(cur_x), float(x2), float(x3), float(x4)],[float(cur_y), float(y2), float(y3), float(y4)]]))
            cur_path += 'C ' + str(x2) + ' ' + str(y2) + ' '+ str(x3) + ' ' + str(y3) + ' ' + str(x4) + ' ' + str(y4) + ' '
            cur_x = x4
            cur_y = y4
            new_paths.append(cur_path)

    return new_paths, new_paths_lengths

def parse_svg_abs(svg_c):
    path_ = svg_c.split('d="')[1]
    path = path_.split('" fill=')[0]
    path_splited = re.split(r"([MLC])", path)
    new_paths = []
    new_paths_lengths = []
    cur_x = 0.0
    cur_y = 0.0
    for idx in range(0,len(path_splited)):
        if len(path_splited[idx]) == 0: continue
        if path_splited[idx] == 'M':
            coords_str = path_splited[idx+1]
            x4 = coords_str.split(' ')[1]
            y4 = coords_str.split(' ')[2]
            cur_x = x4
            cur_y = y4

        if path_splited[idx] == 'L':
            cur_path = 'M ' + cur_x + ' ' + cur_y + ' '
            coords_str = path_splited[idx+1]
            x4 = coords_str.split(' ')[1]
            y4 = coords_str.split(' ')[2]
            new_paths_lengths.append(cal_bezier_length(1, [[float(cur_x), float(x4)],[float(cur_y), float(y4)]]))
            cur_path += 'L ' + x4 + ' ' + y4 + ' '
            cur_x = x4
            cur_y = y4
            new_paths.append(cur_path)
            
        if path_splited[idx] == 'C':
            cur_path = 'M ' + cur_x + ' ' + cur_y + ' '
            coords_str = path_splited[idx+1]
            x2 = coords_str.split(' ')[1]
            y2 = coords_str.split(' ')[2]
            x3 = coords_str.split(' ')[3]
            y3 = coords_str.split(' ')[4]
            x4 = coords_str.split(' ')[5]
            y4 = coords_str.split(' ')[6]
            new_paths_lengths.append(cal_bezier_length(3, [[float(cur_x), float(x2), float(x3), float(x4)],[float(cur_y), float(y2), float(y3), float(y4)]]))
            cur_path += 'C ' + x2 + ' ' + y2 + ' '+ x3 + ' ' + y3 + ' ' + x4 + ' ' + y4 + ' '
            cur_x = x4
            cur_y = y4
            new_paths.append(cur_path)
    return new_paths, new_paths_lengths

fontid_list = ['02', '12', '41']
for fontid in fontid_list:
    if not os.path.exists('./font_' + fontid):
        os.mkdir('./font_' + fontid)
    for data_split in {'syn', 'gt'}:
        input_dir = './font_' + fontid + '_raw/' + data_split + '/'
        out_dir = 'font_' + fontid + '/' + data_split + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for idx in range(0,52):
            svg_f = open(input_dir + "%02d"%(idx) + '.svg')
            svg_c = svg_f.read()
            if data_split == 'syn':
                new_paths, new_paths_lengths = parse_svg_abs(svg_c)
            else:
                new_paths, new_paths_lengths = parse_svg_rel(svg_c)
            if 'fill="currentColor"/>' in svg_c:
                svg_c = svg_c.replace('fill="currentColor"/>','fill="none" stroke="black" stroke-width="0.3"></path>')
            d = svg_c.split('path d=')[1]
            d = 'd=' + d
            d = d.split('</path>')[0]
            fout = open(out_dir + fontid + '_%02d'%(idx) + '.svg', 'w')
            fout.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="50px" height="50px" viewBox="0 0 24 24">' + '\n')
            # define style css
            fout.write('<style type="text/css">' + '\n')
            fout.write('.pen {' + '\n')
            fout.write('\t' + 'stroke-dashoffset: 0;' + '\n')
            fout.write('\t' + 'animation-duration: 10s;' + '\n')
            fout.write('\t' + 'animation-iteration-count: 1000;' + '\n')
            fout.write('\t' + 'animation-timing-function: ease;' + '\n')
            fout.write('}' + '\n')

            length_full = np.sum(new_paths_lengths)
    
            for path_num in range(len(new_paths)):
                fout.write('.path' + '%02d'%path_num + ' {' + '\n')
                fout.write('\t\t' + 'stroke-dasharray: ' + str(max(int(new_paths_lengths[path_num]*2),1)) + ';' + '\n')
                fout.write('\t\t' + 'animation-name: dash' + '%02d '%path_num + '\n')
                fout.write('}' + '\n')

                fout.write('@keyframes dash' + '%02d'%path_num +' {' + '\n')
                
                prop = float(100 * np.sum(new_paths_lengths[0:path_num]) / np.sum(new_paths_lengths))
                if path_num == 0:
                    fout.write('\t' + str(prop) + '% {' + '\n')
                else:
                    fout.write('\t0%, ' + str(prop) + '% {' + '\n')
                fout.write('\t\t' + 'stroke-dashoffset: ' + str(max(int(new_paths_lengths[path_num]*2),1)) + ';' + '\n')
                fout.write('\t' + '}' + '\n')

                prop = float(100 * np.sum(new_paths_lengths[0:path_num+1]) / np.sum(new_paths_lengths))
                fout.write('\t' + str(prop) + '% {' + '\n')
                fout.write('\t\t' + 'stroke-dashoffset: 0;' + '\n')
                fout.write('\t' + '}' + '\n')
                fout.write('}' + '\n')
            fout.write('</style>' + '\n')

            # write paths
            for path_num in range(len(new_paths)):
                fout.write('<path class="pen' + ' path' + '%02d'%path_num + '" ')
                fout.write('d="')
                fout.write(new_paths[path_num])
                fout.write('" fill="none" stroke="black" stroke-width="0.3">')
                #fout.write(d + '\n')
                fout.write('</path>' + '\n')

            fout.write('</svg>' + '\n')
            fout.close()