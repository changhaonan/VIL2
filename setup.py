from setuptools import setup

setup(
   name='vil2',
   version='1.0',
   description='A useful module',
   author='haonan chang',
   author_email='chnme40cs@gmail.com',
   packages=['vil2'],  #same as name
   install_requires=['numpy', 'h5py', 'open3d', 'opencv-python', 'albumentations', 'python-box', 'lightning', 'wandb'], #external packages as dependencies
)