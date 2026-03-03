from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .' # end of requirements.txt for setup.py
def get_requirements(file_path:str)->List[str]: #file_path is str, returns a List
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines() # read the line
        requirements=[req.replace("\n","") for req in requirements] # remove the new line \n

        if HYPEN_E_DOT in requirements: # remove the hypen_e_dot
            requirements.remove(HYPEN_E_DOT) 
    
    return requirements

setup(
name='Chess-Error-Analysis-Prediction',
version='0.0.1',
author='Steven',
author_email='120138752+stevenhan777@users.noreply.github.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)