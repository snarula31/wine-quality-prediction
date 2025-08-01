from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
    
        # If -e . is present, then remove it
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements


setup(
    name='Wine Quality Prediction',
    version='0.0.1',
    author='Shivansh',
    author_email='shivanshnarula@gmail.com',
    pacjakges=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)