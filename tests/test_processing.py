"""
Test the functions in read_and_process.py
"""
import pytest
import os
from src.read_and_process import read_image



def test_analyse_gee_data():

    filename = os.path.join('tests','test_images','RDK2_17_Dc_Pc_Lc.png')
    image_array = read_image(filename)

    filename_tif = os.path.join('tests','test_images','2005_Erps-Kwerps-Villershof.tif')
    image_array_tif = read_image(filename_tif)

    assert image_array.shape==(1595,1465)
    assert image_array_tif.shape==(445,1548)
