#!/usr/bin/env python3
"""Downloads pictures and dates from the digital library mv
"""


import os
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET

download_pictures = False
download_dates = True # to download dates, either low res or high res must be True, but download pictures can be False

low_res = True
high_res = False
name_list = []
if low_res: # download pictures in 800 width
    name_list.append('pictures_all_low_res')
if high_res:   # download pictures in master size
    name_list.append('pictures_all')

 
METS_FULL_TEXT = [
    'https://www.digitale-bibliothek-mv.de/viewer/metsresolver?id=PPN636776093_1910',
    'https://www.digitale-bibliothek-mv.de/viewer/metsresolver?id=PPN636776093_1915',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1916',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1917',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1918',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1919',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1920',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1921',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1922',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1923',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1924',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1925',
    'https://www.digitale-bibliothek-mv.de/viewer/metsresolver?id=PPN636776093_1926',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1927',
    'https://www.digitale-bibliothek-mv.de/viewer/metsresolver?id=PPN636776093_1928',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1929',
    'https://www.digitale-bibliothek-mv.de/viewer/sourcefile?id=PPN636776093_1932'
]



dates = {}


for name in name_list:
    output_path = os.path.join('..','data','000_Raw_Images',name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    for url in METS_FULL_TEXT:
        year = re.sub(r".*_", '', url)
        target_folder = os.path.join(output_path, year)
        abs_target_folder = os.path.join(os.getcwd(), target_folder)
    
        if not os.path.isdir(abs_target_folder):
            os.mkdir(abs_target_folder)
    
        with urlopen(url) as conn:
            data = ET.fromstring(conn.read())
            for fileSec in data.findall('{http://www.loc.gov/METS/}fileSec'):
                for fileGrp in fileSec:
                    if fileGrp.attrib['USE'] != 'DEFAULT':
                        continue
    
                    count = 1
                    for file in fileGrp:
                        file_id = file.attrib['ID']
    
                        for structMap in data.findall('{http://www.loc.gov/METS/}structMap'):
                            if structMap.attrib['TYPE'] != 'PHYSICAL':
                                continue
                            for div in structMap.findall('{http://www.loc.gov/METS/}div'):
                                if div.attrib['TYPE'] == 'physSequence':
                                    for div2 in div.findall('{http://www.loc.gov/METS/}div'):
                                        div2_id = None
                                        for fptr in div2.findall('{http://www.loc.gov/METS/}fptr'):
                                            if fptr.attrib['FILEID'] == file_id:
                                                div2_id = div2.attrib['ID']
                                                break
    
                                        for structLink in data.findall('{http://www.loc.gov/METS/}structLink'):
                                            for smLink in structLink.findall('{http://www.loc.gov/METS/}smLink'):
                                                if smLink.attrib['{http://www.w3.org/1999/xlink}to'] == div2_id:
    
                                                    for structMap2 in data.findall('{http://www.loc.gov/METS/}structMap'):
                                                        if structMap2.attrib['TYPE'] != 'LOGICAL':
                                                            continue
                                                        for div3 in structMap2.findall('{http://www.loc.gov/METS/}div'):
    
                                                                for div4 in div3.findall('{http://www.loc.gov/METS/}div'):
                                                                        for div5 in div4.findall('{http://www.loc.gov/METS/}div'):
                                                                            if div5.attrib['ID'] == smLink.attrib['{http://www.w3.org/1999/xlink}from']:
                                                                                if div5.attrib['TYPE'] == 'cover':
                                                                                    date_issued = ""
                                                                                    continue
                                                                                log_id = div5.attrib['DMDID']
    
                                                                                for dmdSec in data.findall('{http://www.loc.gov/METS/}dmdSec'):
                                                                                    if dmdSec.attrib['ID'] == log_id:
                                                                                        for originInfo in dmdSec[0][0][0].findall(
                                                                                                '{http://www.loc.gov/mods/v3}originInfo'):
                                                                                            for dateIssued in originInfo.findall(
                                                                                                    '{http://www.loc.gov/mods/v3}dateIssued'):
                                                                                                date_issued = dateIssued.text
    
                        for Flocat in file:
                            image_url_comp = Flocat.attrib['{http://www.w3.org/1999/xlink}href']
                            image_url_full = image_url_comp.replace('/viewer/content/', '/viewer/api/v1/records/')
                            image_url_full = image_url_full.replace("/800/0/", "/files/images/")
                            image_url_full = image_url_full.replace('.jpg', '.tif')
                            if name == 'pictures_all_low_res':
                                image_url_full = image_url_full + '/full/800,/0/default.jpg'
                            else:
                                image_url_full = image_url_full+'/full/!3785,5687/0/default.jpg'
                            print('Downloading %s ...' % (image_url_full))
    
    
    
                            filename = re.sub(r".*/", "", image_url_comp)
    
                            # rename pictures from 1915 to make numbering consistent
                            year = url.split('_')[-1]
                            if year == '1915':
                                filename = '{0:08d}'.format(count) + '.jpg'
                                count += 1
    
                            filepath = os.path.join(abs_target_folder, filename)
                            dates[os.path.join(target_folder, filename)] = date_issued
    
                            if download_pictures:
                                try:
                                    with urlopen(image_url_full) as fulltext_conn:
                                        with open(filepath, 'b+w') as f:
                                            f.write(fulltext_conn.read())
                                except Exception:
                                    print("ERROR")



if download_dates:
    os.makedirs('../data/meta_data/dates', exist_ok=True)
    years = [tmp.split('_')[-1] for tmp in METS_FULL_TEXT]
    for year in years:
        os.makedirs(os.path.join('../data/meta_data/dates', year), exist_ok=True)
        year_list = [(key,value) for key,value in dates.items() if year in key]

        with open(os.path.join('../data/meta_data/dates', year, 'file_date_connection.txt'), 'w') as file:
            file.write('file \t date\n')
            for key, value in year_list:
                file.write('%s\t%s\n' % (key.replace(name+'/',''), value))