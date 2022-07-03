import pydiffvg
import argparse
import torch
import skimage.io
import os
import re
from shutil import copyfile
import shutil
from PIL import Image
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

gamma = 1.0

def cal_alignment_loss(args, save_path):
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW

    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    points_vars = []
    for path in shapes:
        #print(path)
        #input()
        path.points.requires_grad = True
        points_vars.append(path.points)
    color_vars = {}
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars[group.fill_color.data_ptr()] = group.fill_color
    color_vars = list(color_vars.values())

    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1)
    color_optim = torch.optim.Adam(color_vars, lr=0)

    # Adam iterations.
    for t in range(args.num_iter):
        
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     0,   # seed
                     None, # bg
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        loss = (img - target).pow(2).mean()
        #if t%10 == 0:
        #    print('iteration:', t)
        #    print('render loss:', args.no_sample, loss.item())
    
        # Backpropagate the gradients.
        loss.backward()
    
        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        if t == args.num_iter - 1:

            pydiffvg.save_svg_paths_only(save_path, canvas_width, canvas_height, shapes, shape_groups)
        
    return loss

def get_svg_glyph_bbox(svg_path):
    fin = open(svg_path,'r')
    path_ = fin.read().split('d="')[1]
    path = path_.split('" fill=')[0]
    path_splited = re.split(r"([mlc])", path)
    commands = []
    cur_x = 0.0
    cur_y = 0.0
    x_min = 1000
    x_max = -1000
    y_min = 1000
    y_max = -1000
    first_move = True
    for idx in range(0,len(path_splited)):
        if len(path_splited[idx]) == 0: continue
        # x1,y1,x2,y2,x3,y3,x4,y4 are the absolute coords
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
            x_min = min(cur_x, x_min)
            x_max = max(cur_x, x_max)
            y_min = min(cur_y, y_min)
            y_max = max(cur_y, y_max)
        if path_splited[idx] == 'l':
            coords_str = path_splited[idx+1]
            x4 = cur_x + float(coords_str.split(' ')[1])
            y4 = cur_y + float(coords_str.split(' ')[2])
            cur_x = x4
            cur_y = y4
            x_min = min(cur_x, x_min)
            x_max = max(cur_x, x_max)
            y_min = min(cur_y, y_min)
            y_max = max(cur_y, y_max)
        if path_splited[idx] == 'c':
            coords_str = path_splited[idx+1]
            x1 = cur_x
            y1 = cur_y
            x2 = cur_x + float(coords_str.split(' ')[1])
            y2 = cur_y + float(coords_str.split(' ')[2])
            x3 = cur_x + float(coords_str.split(' ')[3])
            y3 = cur_y + float(coords_str.split(' ')[4])
            x4 = cur_x + float(coords_str.split(' ')[5])
            y4 = cur_y + float(coords_str.split(' ')[6])
            x_min = min(x2, x3, x4, x_min)
            x_max = max(x2, x3, x4, x_max)
            y_min = min(y2, y3, y4, y_min)
            y_max = max(y2, y3, y4, y_max)
            cur_x = x4
            cur_y = y4
    return [x_min,x_max], [y_min,y_max]

def get_img_bbox(img_path):
    img = Image.open(img_path)
    img = 255 - np.array(img)
    img0 = np.sum(img, axis=0)
    img1 = np.sum(img, axis=1)
    y_range = np.where(img1>127.5)[0]
    x_range = np.where(img0>127.5)[0]
    return [x_range[0], x_range[-1]], [y_range[0], y_range[-1]]


def trans_svg_w_align2img(svg_path, trgimg_path):

    svg_xr, svg_yr = get_svg_glyph_bbox(svg_path)
    img_xr, img_yr = get_img_bbox(trgimg_path)
    svg_w = svg_xr[1] - svg_xr[0]
    svg_h = svg_yr[1] - svg_yr[0]
    svg_xc = (svg_xr[1] + svg_xr[0]) / 2.0
    svg_yc = (svg_yr[1] + svg_yr[0]) / 2.0
    img_w = img_xr[1] - img_xr[0] + 1
    img_h = img_yr[1] - img_yr[0] + 1
    img_xc = (img_xr[1] + img_xr[0]) / 2.0
    img_yc = (img_yr[1] + img_yr[0]) / 2.0    

    def affine_coord(coord, x_or_y, cur_cmd, first_move):
        
        if x_or_y % 2 == 0: # for x
            if cur_cmd == 'm' and first_move:
                new_coord = (coord - svg_xc) * (img_w / svg_w) + img_xc
                res = str(new_coord)
            else:
                res = str((img_w / svg_w) * (coord))
        else: # for y 
            if cur_cmd == 'm' and first_move:
                new_coord = (coord - svg_yc) * (img_h / svg_h) + img_yc
                res = str(new_coord)
            else:
                res = str((img_h / svg_h) * (coord))
    
        return res

    svg_raw = open(svg_path,'r').read()
    fout = open(svg_path.split('.svg')[0] + '_256.svg','w')
    fout.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="256px" height="256px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 256">')
    
    coord = '<path' + svg_raw.split('<path')[1]
    tokens = coord.split(' ')
    newcoord = ''
    first_move = True
    x_or_y = 0
    for k in tokens:
        if k[0] != '<' and k[0] != 'd' and k[0] != 'm' and k[0] != 'c' and k[0] != 'l' and k[0] != 'f':
            if k[-1] != '"':
                newcoord += affine_coord(float(k), x_or_y, cur_cmd, first_move)
                if cur_cmd == 'm': first_move = False
                x_or_y += 1
                newcoord += ' '
            else:
                newcoord += affine_coord(float(k[0:len(k)-1]), x_or_y, cur_cmd, first_move)
                x_or_y += 1                             
                newcoord += '" '
        else:
            cur_cmd = k
            newcoord += k
            newcoord += ' '
    fout.write(newcoord)
    fout.close()


def trans_svg_wo_align2img(svg_path, trgimg_path):

    svg_raw = open(svg_path,'r').read()
    fout = open(svg_path.split('.svg')[0] + '_256.svg','w')
    fout.write('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="256px" height="256px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 256">')
    scalar = 256 / 24
    coord = '<path' + svg_raw.split('<path')[1]
    tokens = coord.split(' ')
    newcoord = ''
    for k in tokens:
        if k[0] != '<' and k[0] != 'd' and k[0] != 'm' and k[0] != 'c' and k[0] != 'l' and k[0] != 'f':
            if k[-1] != '"':
                newcoord += str(float(k) * scalar)
                newcoord += ' '
            else:
                newcoord += str(float(k[0:len(k)-1]) * scalar)
                newcoord += '" '
        else:
            newcoord += k
            newcoord += ' '

    fout.write(newcoord)
    fout.close()

def process_s1(process_id, chars_per_process, args):

    svg_path = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'svgs')
    imghr_path = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'imgs_256')
    svg_outpath = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'svgs_bestcand')
    if not os.path.exists(svg_outpath):
        os.mkdir(svg_outpath)

    for i in range(process_id * chars_per_process, (process_id + 1) * chars_per_process):
        if i >= args.num_chars:
            break
        # find the best candidate
        minLoss = 10000
        noMin = 0
        tempLoss = 0
        # pick the best candidate
        for j in range(0, int(args.candidate_nums)):
            args.no_sample = j
            args.svg = os.path.join(svg_path, 'syn_%02d_%02d.svg'%(i,j))
            args.target = os.path.join(imghr_path, '%02d_256.png'%i)
            if args.init_svgbbox_align2img:
                trans_svg_w_align2img(args.svg, args.target)
            else:
                trans_svg_wo_align2img(args.svg, args.target)
            args.svg = os.path.join(svg_path, 'syn_%02d_%02d_256.svg'%(i,j))
            tempLoss = cal_alignment_loss(args, save_path = args.svg.split('.svg')[0] + '_r.svg')
            if tempLoss < minLoss:
                noMin = j
                minLoss = tempLoss
        # do longer optimization
        src_path = os.path.join(svg_path, 'syn_%02d_%02d_256.svg'%(i,noMin))
        trg_path = os.path.join(svg_outpath, 'syn_%02d_256.svg'%(i))
        shutil.copy(src_path, trg_path)
 
def process_s2(process_id, chars_per_process, args):
    imghr_path = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'imgs_256')
    svg_path = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'svgs')
    svg_cdt_path = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'svgs_bestcand')
    svg_outpath = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'svgs_refined')
    if not os.path.exists(svg_outpath):
        os.mkdir(svg_outpath)
    
    for i in range(process_id * chars_per_process, (process_id + 1) * chars_per_process):
        if i >= args.num_chars:
            break
        # refine the best candidate
        args.num_iter = 300
        args.svg = os.path.join(svg_cdt_path, 'syn_%02d_256.svg'%(i))
        args.target = os.path.join(imghr_path, '%02d_256.png'%i)
        tempLoss = cal_alignment_loss(args, save_path = os.path.join(svg_outpath, 'syn_%02d.svg'%(i)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", help="source SVG path", type=str, default='none')
    parser.add_argument("--target", help="target image path", type=str, default='none')
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=40)
    parser.add_argument("--no_sample", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_chars", type=int, default=52)
    parser.add_argument("--fontid", type=str, default='17')
    parser.add_argument("--experiment_name", type=str, default='dvf')
    parser.add_argument("--candidate_nums", type=str, default='20')
    parser.add_argument("--init_svgbbox_align2img", type=bool, default=False)
    args = parser.parse_args()

    svg_outpath = os.path.join('experiments', args.experiment_name + '_main_model/results/', '%04d'%int(args.fontid), 'svgs_refined')
    
    chars_per_process = args.num_chars // args.num_processes + 1
    
    print("stage 1: find the best candidates ...")
    processes = [mp.Process(target=process_s1, args=[pid, chars_per_process, args]) for pid in range(args.num_processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
          
    print("stage 2: further refine these candidates ...")
    processes = [mp.Process(target=process_s2, args=[pid,chars_per_process, args]) for pid in range(args.num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    svg_merge_outpath = os.path.join(svg_outpath, f"syn_svg_merge.html")
    fout = open(svg_merge_outpath, 'w')
    for i in range(0, 52):
        svg = open(os.path.join(svg_outpath, 'syn_%02d.svg'%(i)),'r').read()
        svg = svg.replace('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="256" height="256">', '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="64px" height="64px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 256">')
        fout.write(svg)
        if i > 0 and i % 13 == 12:
            fout.write('<br>')
    fout.close()
