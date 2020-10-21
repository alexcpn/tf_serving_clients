"""
###############################
#  Generic TF Serving Client  #
###############################
"""
from setuptools import setup, find_packages


setup(name='tfservingclient',
      packages=find_packages(exclude=["test", "test.*"]),
      version="1.0.0",
      description='Generic TF Serving Client',
      author='Alex Punnen',
      author_email='alexcpn@gmail.com',
      license='Apache 2 License',
      include_package_data=True
      )