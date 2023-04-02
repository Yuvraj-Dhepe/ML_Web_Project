from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    file = open(file_path,'r')
    
    for line in file:
        if "-e ." not in line:
            requirements.append(line.strip('\n'))
    file.close()
    
    #print(requirements)
    return requirements
    
    
setup(
    name='mlproject',
    version='0.0.1',
    author='Yuvraj',
    author_email='yuvi.kiit@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)