import xml.etree.ElementTree as ET
import os
from progress.bar import Bar

def main():
  annotation_dir = 'data/imagenet/ILSVRC/Annotations/CLS-LOC/val/'
  data_dir = 'data/imagenet/ILSVRC/Data/CLS-LOC/val/'
  destination_dir = 'data/imagenet/ILSVRC/Data/CLS-LOC/val_as_test/'
  
  if not os.path.exists(destination_dir):
    print('Preparing ImageNet test set for PyTorch image file')
    os.system(f'mkdir {destination_dir}')
    
    bar = Bar('Processing', max = 50000, suffix = '%(index)d/%(max)d - %(elapsed)ds')
    for i in range (50000):
      tree = ET.parse(annotation_dir + f'ILSVRC2012_val_{(i+1):08d}.xml')
      root = tree.getroot()
      
      if not os.path.exists(destination_dir + root[5][0].text):
        os.system(f'mkdir {destination_dir + root[5][0].text}')
      
      os.system(f"cp {data_dir}ILSVRC2012_val_{(i+1):08d}.JPEG {destination_dir + root[5][0].text}")
      bar.next()
    bar.finish()
  else:
    print('ImageNet test set already prepared for PyTorch image file!')      


if __name__ == "__main__":
    main()