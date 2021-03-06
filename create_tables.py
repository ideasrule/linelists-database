#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:49:14 2019

@author: toma
"""
#this file creates linelist database and tables in mysql

from connection_info import db_url, db_user, db_passwd, db_name
import MySQLdb
from query_functions import sql_order
import time

###############

def create_database():
    db = MySQLdb.connect(host=db_url, user=db_user, passwd=db_passwd)
    cursor = db.cursor()
    cursor.execute('DROP DATABASE IF EXISTS {}'.format(db_name))
    cursor.execute('CREATE DATABASE {}'.format(db_name))
    db.commit()
    cursor.close()
    db.close()

##################

#molecule_name format for example, CO2, is (13C)(16O)2
particles_table_create_query = "CREATE TABLE IF NOT EXISTS particles (\
molecule_name VARCHAR(5) NOT NULL, \
iso_name VARCHAR(25) NOT NULL, \
iso_abundance DOUBLE NOT NULL, \
iso_mass DOUBLE NOT NULL, \
default_line_source_id \
SMALLINT NOT NULL, \
particle_id SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY\
);"

#create table for all the lines for each particle in table particles
#nu stands for transition wavenumber
#a stands for einstein coefficient
#g_upper stands for the degeneracy of the uppper state
transitions_table_create_query = \
"CREATE TABLE IF NOT EXISTS transitions (\
nu DOUBLE NOT NULL, A FLOAT NOT NULL, \
gamma_air FLOAT, \
n_air FLOAT, \
delta_air FLOAT, \
elower DOUBLE NOT NULL, \
g_upper SMALLINT NOT NULL, \
gamma_H2 FLOAT, \
n_H2 FLOAT, \
delta_H2 FLOAT, \
gamma_He FLOAT, \
n_He FLOAT, \
delta_He FLOAT, \
line_source_id SMALLINT UNSIGNED NOT NULL, \
particle_id SMALLINT UNSIGNED NOT NULL, \
line_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, \
FOREIGN KEY (particle_id) REFERENCES particles(particle_id) ON UPDATE CASCADE ON DELETE CASCADE, \
FOREIGN KEY (line_source_id) REFERENCES source_properties(line_source_id) ON UPDATE CASCADE ON DELETE CASCADE) \
ROW_FORMAT=COMPRESSED;"

#create table for the partition coefficient across all temperatures for each particle in table 1
partitions_table_create_query = \
"CREATE TABLE IF NOT EXISTS partitions (\
temperature FLOAT NOT NULL, \
`partition` FLOAT NOT NULL, \
line_source_id SMALLINT UNSIGNED NOT NULL, \
particle_id SMALLINT UNSIGNED NOT NULL, \
partition_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, \
FOREIGN KEY (particle_id) REFERENCES particles(particle_id) ON UPDATE CASCADE ON DELETE CASCADE, \
FOREIGN KEY (line_source_id) REFERENCES source_properties(line_source_id) ON UPDATE CASCADE ON DELETE CASCADE\
);" 

#create table source_properties to store the limits and availbility of the parameters for each source from HITRAN or EXOMOL
source_properties_table_create_query = "CREATE TABLE IF NOT EXISTS source_properties (\
line_source VARCHAR(25) NOT NULL, \
max_temperature SMALLINT, \
max_nu DOUBLE, \
num_lines BIGINT, \
bool_air ENUM('YES', 'NO'), \
bool_H2 ENUM('YES', 'NO'), \
bool_He ENUM('YES', 'NO'), \
reference_link VARCHAR(250) NOT NULL, \
particle_id  SMALLINT UNSIGNED NOT NULL, \
line_source_id SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, \
FOREIGN KEY (particle_id) REFERENCES particles(particle_id) ON UPDATE CASCADE ON DELETE CASCADE);"

#create a view table for user to look through
default_linelists_table_view_create_query = 'CREATE VIEW default_linelists AS \
SELECT p.molecule_name AS molecule, p.iso_name AS isotopologue, p.iso_abundance AS iso_abundance, p.iso_mass AS iso_mass, \
t.nu AS nu, t.A AS A, t.elower AS elower, t.g_upper AS g_upper, t.gamma_air AS gamma_air, \
t.gamma_H2 AS gamma_H2, t.gamma_He AS gamma_He, t.n_air AS n_air, t.n_H2 AS n_H2, t.n_He AS n_He, \
t.delta_air AS delta_air, t.delta_H2 AS delta_H2, t.delta_He AS delta_He, s.line_source AS source \
FROM particles AS p INNER JOIN transitions AS t ON p.particle_id = t.particle_id \
INNER JOIN source_properties AS s ON (p.particle_id = s.particle_id AND p.default_line_source_id = s.line_source_id) ORDER BY nu;'

##################

#create indexes on nu, A, elower, and line_source in table transitions
create_index_line_source_id = "CREATE INDEX line_source_index ON transitions(line_source_id) USING BTREE;"
create_index_nu = "CREATE INDEX nu_index ON transitions(nu) USING BTREE;"
create_index_A = "CREATE INDEX A_index ON transitions(A) USING BTREE;"
create_index_elower = "CREATE INDEX elower_index ON transitions(elower) USING BTREE;"

##################
        
def create_database_with_tables_and_indexes():
    
    start_time = time.time()
    '''
    #create the database first and drop it if it exists already
    create_database()
    
    #create the tables
    sql_order(particles_table_create_query)
    sql_order(source_properties_table_create_query)
    sql_order(transitions_table_create_query)
    sql_order(partitions_table_create_query)
    '''
    
    #Finished line source id index in 239.55348300933838 seconds
    #Finished nu index in 407.1092050075531 seconds
    #Finished A index in 332.6726574897766 seconds
    #Finished elower index in 459.4686324596405 seconds
    #Finished in 1438.8043451309204 seconds
    #for total 137802916 lines
    
    '''
    #create the indexes in table transitions
    #index significance: line_source_id > nu > A > elower
    #yet line_source_index is not needed since foreign key is already present
    #only need to recreate nu index, A index, and line_source_foreign_key
    print('starting...')
    t1 = time.time()
    sql_order('ALTER TABLE transitions ADD CONSTRAINT `transitions_ibfk_2` FOREIGN KEY (line_source_id) \
              REFERENCES source_properties(line_source_id) ON UPDATE CASCADE ON DELETE CASCADE')
    print("Finished line source foreign key in transitions in %s seconds" % (time.time() - t1))
    #t1 = time.time()
    #sql_order(create_index_line_source_id)
    #print("Finished line source id index in %s seconds" % (time.time() - t1))
    t2 = time.time()
    sql_order(create_index_nu)
    print("Finished nu index in %s seconds" % (time.time() - t2))
    t3 = time.time()
    sql_order(create_index_A)
    print("Finished A index in %s seconds" % (time.time() - t3))
    #diable elower index...insert later if needed
    #t4 = time.time()
    #sql_order(create_index_elower)
    #print("Finished elower index in %s seconds" % (time.time() - t4))
    '''
    
    '''
    #used to drop the indexes
    #index significance: line_source_id > nu > A > elower
    print('starting...')
    t1 = time.time()
    sql_order('DROP INDEX `A_index` ON transitions;')
    print("Finished A index index in %s seconds" % (time.time() - t1))
    t2 = time.time()
    sql_order('DROP INDEX `nu_index` ON transitions;')
    print("Finished nu index in %s seconds" % (time.time() - t2))
    t3 = time.time()
    sql_order('DROP INDEX `line_source_index` ON transitions;')
    print("Finished line source index in %s seconds" % (time.time() - t3))
    '''
    print("Finished in %s seconds" % (time.time() - start_time))
        
if __name__ == '__main__':
    create_database_with_tables_and_indexes()
