import argparse
import os
import csv
from PIL import Image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_folder', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    parser_args = parser.parse_args()

    pdf_path = os.path.join(parser_args.save_folder, 'vis_list.pdf')

    
    
    # Load ordered case id 
    ordered_id = []
    with open(parser_args.csv_path,mode='r',encoding='utf-8',newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'file_name': continue
            ordered_id.append(row[0])
    # print(ordered_id)

    img_open_list = []       
    for root, dirs, files in os.walk(parser_args.vis_folder):
        for idx in ordered_id:
            file = os.path.join(root, idx+'.png')
            img_open = Image.open(file)                
            if img_open.mode != 'RGB':
                img_open = img_open.convert('RGB')     
            img_open_list.append(img_open)
    img_1 = img_open_list[0]

    img_open_list = img_open_list[1:]


    img_1.save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=img_open_list)
    