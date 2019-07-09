#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:27:36 2019

@author: toma
"""

import hapi
import numpy as np
import time
import insert_hitran
from query_functions import sql_order
import os

#this thing gives us the table of molecule properties in shell
#saved it manually in a file called /home/toma/Desktop/molecule_properties.txt
#getHelp(ISO_ID)

start_time = time.time()

'''
#change file to loadable version
with open('/home/toma/Desktop/molecule_properties.txt', 'r') as f: 
    outfile = open('/home/toma/Desktop/molecule_properties (copy).txt', 'w')
    for line in f: 
        data = line.split()
        data.pop(1)
        for i in data:
            outfile.write("%s " % i)
        outfile.write('\n')
    outfile.close()
f.close()
'''    

#everything is in string though to be noticed
#length of lists are 124
mol_ids, iso_ids, iso_names, iso_abundances, iso_masses, mol_names = \
np.loadtxt('/home/toma/Desktop/molecule_properties (copy).txt', dtype='str', skiprows=1, usecols=(1, 2, 3, 4, 5, 6), unpack=True)

'''
for i in range(len(mol_ids)):
    particle_property_query = "INSERT INTO particles VALUES('%s', '%s', '%s', '%s', '%s', null);" % (mol_names[i], iso_names[i], \
                                                           iso_abundances[i], iso_masses[i], 'HITRAN_2016')
    #insert each molecule's properties into particles table
    sql_order(particle_property_query)
    
    #then, fetch all the data from HITRAN using HAPI
    hapi.db_begin('data')
    #becasue cannot choose inifinity as upper limit, use a giant number instead
    hapi.fetch(mol_names[i], mol_ids[i], iso_ids[i], 0, 100000000000000000, ParameterGroups=['Standard', 'Voigt_Air', 'Voigt_H2', 'Voigt_He'], Parameters=['nu',\
               'a', 'gamma_air', 'n_air', 'delta_air', 'elower', 'gp', 'gamma_H2', 'n_H2', 'delta_H2', 'gamma_He', 'n_He', 'delta_He'])\
    
    #open the file and use insert_hitran.py to insert all parameters into transitions table
    filename = '/home/toma/Desktop/linelists-database/data/{}.data'.format()
    insert_hitran.insert_hitran(filename)
    
    #delete the files since the files are named by HAPI using mol_name instead of iso_name
    #s.t. python wont get confused in the for loop
    header_filename = '/home/toma/Desktop/linelists-database/data/{}.header'.format()
    os.remove(filename)
    os.remove(header_filename)
'''   
    

################don't know why this not workkkkkkkkkkkkkkkkkkkk???????????didnt go with specified params
hapi.db_begin('data')
#becasue cannot choose inifinity as upper limit, use a giant number instead
hapi.fetch('CO', 5, 1, 0, 100000000000000000, ParameterGroups=['Standard', 'Voigt_Air', 'Voigt_H2', 'Voigt_He'], Parameters=['nu',\
           'a', 'gamma_air', 'n_air', 'delta_air', 'elower', 'gp', 'gamma_H2', 'n_H2', 'delta_H2', 'gamma_He', 'n_He', 'delta_He'])
    
#if above not work for formatting use this to select params from file
#hapi.select('CO', ParameterNames=('nu', 'a', 'gamma_air', 'n_air', 'delta_air', 'elower', 'gp', 'gamma_H2', 'n_H2', 'delta_H2', 'gamma_He', 'n_He', 'delta_He'))
    
print("Finished in %s seconds" % (time.time() - start_time))