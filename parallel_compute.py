#!/home/stanley/anaconda3/bin/python
from mpi4py import MPI
import compute_absorption
import numpy as np
from query_functions import fetch
import pdb
from connection_info import db_url, db_user, db_passwd, db_name
import MySQLdb
import scipy.special

#store the value of speed of light in cm/s
c = 2.99792458e10
#planck constant erg*s
h = 6.62606885e-27
#boltzmann constant
k_B = 1.38064852e-16
#reference Temperature in K
T_ref = 296 
#store the value of c_2 = h * c / k_B
c_2 = h * c / k_B
#store the conversion from gram to amu
G_TO_AMU = 1.66054e-24#1.66053904e-24
#store pi value
pi = 3.1415926535897932384626433
BAR_TO_PASCAL = 1e5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
T_grid = np.arange(100, 3100, 100)
P_grid = 10.0**np.arange(-4, 9)
#T_grid = np.arange(1000, 3100, 100)
#P_grid = 10.0**np.arange(4, 9)
lambda_grid = np.exp(np.linspace(np.log(0.3e-6), np.log(30e-6), 4616))


#given input v(nu), T, p, iso, source, and version

#fetch the partition function value given an input T, temperature
def get_partition(line_source_id, particle_id): #temp has to be a float i.g. 19.0
    print(line_source_id, particle_id)
    #query for the partition function given T, temperature
    query = "SELECT temperature, `partition` FROM partitions WHERE line_source_id = {} AND particle_id = {}".format(line_source_id, particle_id)
    
    temperatures, Qs = np.array(fetch(query)).T
    
    return temperatures, Qs

#######################

#get the particle id and the isotopologue abundance of the input iso_name 
#iso_name format for example, CO2, is (13C)(16O)2
def get_particle(iso_name):
    query = "SELECT particle_id, iso_abundance, iso_mass FROM particles WHERE iso_name = '{}'".format(iso_name)
    
    data = fetch(query)
    
    if len(data) != 1:
        raise Exception('should have exactly one row for a specific particle')
    
    #data[0] = (particle_id, iso_abundance, iso_mass)
    return data[0]

        
#########################
#@@profile
#@jit(parallel=False, fastmath=True)
def compute_one_wavenum(wavenumber, T, iso_abundance, iso_mass, Q, v_ij_star, a, elower, g_upper, gamma_p_T):#, lower_indexes, upper_indexes):
    #compute line intensity function S_ij(T)
    S_ij = iso_abundance * a * g_upper * np.exp(-c_2 * elower / T) * (1 - np.exp(-c_2 * v_ij_star / T)) / (8 * pi * c * v_ij_star**2 * Q)
    
    #alternative way to compute voigt function: 70.0055468082428 seconds ; 0.01% diff ; 18% speed up
    sigma_thermal = np.sqrt(k_B * T / iso_mass / G_TO_AMU / c**2) * v_ij_star
    z = (wavenumber - v_ij_star + gamma_p_T * 1j) / sigma_thermal / np.sqrt(2)
    wofz = scipy.special.wofz(z)
    voigt_profile = np.real(wofz) / sigma_thermal / np.sqrt(2*pi)
    absorption = S_ij * voigt_profile      
    return np.sum(absorption)
    


def handle_master(iso_name, line_source="default", max_cutoff=1000):
    wavenums = 1e-2 / lambda_grid
    particle_data = get_particle(iso_name)
    particle_id = particle_data[0]
    iso_abundance = particle_data[1]
    iso_mass = particle_data[2]
    print(particle_id, iso_abundance, iso_mass)
   
    #get line source id
    if line_source == 'default': 
        line_source_id = fetch("SELECT default_line_source_id FROM particles WHERE particle_id = {}".format(particle_id))[0][0]
    else: 
        get_line_source_id_query = "SELECT line_source_id FROM source_properties WHERE line_source = '{}' and \
        particle_id = {}".format(line_source, particle_id)
        data = fetch(get_line_source_id_query)
        if len(data) != 1:
            raise Exception('should have exactly one line_source_id corresponding to one line_source and isotopologue')   
        line_source_id = data[0][0]
    print("Using line source ID", line_source_id)


    #if computing using hitemp data, use hitran partitions, so get hitran line_source_id for partitions
    if 'HITEMP' in line_source: 
        get_hitran_source_id_query = "SELECT line_source, line_source_id FROM source_properties WHERE particle_id = {}".format(particle_id)
        sources = fetch(get_hitran_source_id_query)
        hitran_id = -1
        for source in sources: 
            if source[0].startswith('HITRAN'): 
                hitran_id = source[1]
        if hitran_id == -1:
            raise Exception('This isotopologue has hitemp but no hitran linelist which is weird')
        #use hitran id to get partitions for hitemp
        temperatures, Qs = get_partition(hitran_id, particle_id)    
    else: #for other sources, use line source id to get partitions   
        #get paritition using the correct function
        temperatures, Qs = get_partition(line_source_id, particle_id)
        
    min_id, max_id = fetch("SELECT min(line_id), max(line_id) from transitions where line_source_id={} AND nu >= {} AND nu < {}".format(
        line_source_id, min(wavenums) - max_cutoff, max(wavenums) + max_cutoff))[0]
    num_lines = max_id - min_id + 1
    print("Number of lines", num_lines)
    
    for target_rank in range(1, nprocs):
        lines_per_proc = int(round(num_lines / (nprocs - 1)))
        start = min_id + lines_per_proc * (target_rank - 1)
        end = min_id + lines_per_proc * target_rank
        print("rank, start, end", target_rank, start-min_id, end-min_id, 0, max_id-min_id)
        comm.send([iso_abundance, iso_mass, line_source_id, temperatures, Qs, start, end], dest=target_rank)        

    #Collect all the results
    absorption = np.zeros((len(T_grid), len(P_grid), len(lambda_grid)))
    counter = 0
    for P_index in range(len(P_grid)):
        for T_index in range(len(T_grid)):
            for target_rank in range(1, nprocs):
                data = comm.recv(source=target_rank, tag=counter)
                absorption[T_index, P_index] += data
                if P_index == 8 and T_index == 9:
                    print("I get this at 3616:", absorption[T_index, P_index, 3616])
            np.save("absorb_T{}_logP{}.npy".format(T_grid[T_index], int(np.log10(P_grid[P_index]))), absorption[T_index, P_index])
            counter += 1
    return absorption
    pdb.set_trace()

def handle_slave():
    wavenums = 1e-2 / lambda_grid#[::-1]
    #pdb.set_trace()
    iso_abundance, iso_mass, line_source_id, Q_temperatures, Qs, start, end = comm.recv(source=0)

    query = "SELECT nu, A, gamma_air, n_air, delta_air, elower, g_upper, gamma_H2, \
    n_H2, delta_H2, gamma_He, n_He, delta_He FROM transitions WHERE \
    line_source_id = '{}' AND line_id >= {} AND line_id < {} order by nu".format(line_source_id, start, end)

    db = MySQLdb.connect(host=db_url, user=db_user, passwd=db_passwd, db=db_name)
    cursor = db.cursor()
    cursor.execute(query)

    lines_table = cursor.fetchall()
    lines_array = np.asarray(lines_table, dtype=np.float64)
    print(lines_array.shape)

    ###############construct gamma and n arrays
    #assuming cond returns an array of indexes that are within the range of cut off
    #parameters for fetchall() data corresponding to tuple indexes: 
    #(nu, a, gamma_air, n_air, delta_air, elower, g_upper, gamma_H2, n_H2, delta_H2, gamma_He, n_He, delta_He)
    #(0,  1,     2,       3,       4,        5,     6,    7,       8,      9,      10,      11,     12   )

    
    v_ij = lines_array[:,0] #v_ij = nu !!!!!!!
    print(v_ij)
    a = lines_array[:,1]
    gamma_air = lines_array[:,2]
    n_air = lines_array[:,3]
    delta_air = lines_array[:,4]
    elower = lines_array[:,5]
    g_upper = lines_array[:,6]
    gamma_H2 = lines_array[:,7]
    n_H2 = lines_array[:,8]
    delta_H2 = lines_array[:,9]
    gamma_He = lines_array[:,10]
    n_He = lines_array[:,11]
    delta_He = lines_array[:,12]

    #################
    #initialize an np array for gamma_p_T
    
    #arrays that check whether that parameter is null in that index
    bool_gamma_H2 = ~np.isnan(gamma_H2)
    bool_gamma_He = ~np.isnan(gamma_He)
    bool_n_H2 = ~np.isnan(n_H2)
    bool_n_He = ~np.isnan(n_He)

    #where f_H2 = 0.85 and f_He = 0.15
    #if either n_H2 or n_He does not exist, f_H2/He (the exisiting one) = 1.0
    has_H2_and_He_gamma_N = np.all([bool_gamma_H2, bool_n_H2, bool_gamma_He, bool_n_He], axis=0)
    
    #if n_H2 does not exist, f_He = 1
    has_He_but_not_H2_gamma_N = np.all([bool_gamma_He, bool_n_He, ~np.logical_or(bool_gamma_H2, bool_n_H2)], axis=0)
   
    #if n_He does not exist, f_H2 = 1
    has_H2_but_not_He_gamma_N = np.all([bool_gamma_H2, bool_n_H2, ~np.logical_or(bool_gamma_He, bool_n_He)], axis=0)
   
    #if both n_H2 or n_He does not exist
    has_only_air_gamma_N = np.all([~bool_gamma_H2, ~bool_n_H2, ~bool_gamma_He, ~bool_n_He], axis=0)
   
    #arrays that check whether that parameter is null in that index
    bool_delta_H2 = ~np.isnan(delta_H2)
    bool_delta_He = ~np.isnan(delta_He)
    bool_delta_air = ~np.isnan(delta_air)

    #compute v_ij_star for f 
    #v_ij_star = v_ij + delta_net * p, wcounter += 1
    #here delta_net is computed in similar fashion to gamma_p_T
    has_H2_and_He_delta = np.logical_and(bool_delta_H2, bool_delta_He)
   
    #when delta_H2 does not exist, f_He = 1.0
    has_He_but_not_H2_delta = np.logical_and(~bool_delta_H2, bool_delta_He)
    
    #when delta_He does not exist, f_H2 = 1.0
    has_H2_but_not_He_delta = np.logical_and(bool_delta_H2, ~bool_delta_He)    

    #when both delta_H2 and delta_He does not exist, use delta_air
    has_air_but_not_H2_and_He_delta = np.all([bool_delta_air, ~bool_delta_H2, ~bool_delta_He], axis=0)    

    #when all deltas do not exist
    has_no_delta = np.all([~bool_delta_air, ~bool_delta_H2, ~bool_delta_He], axis=0)   
    #need to pass in: v_ij_star, a, elower, g_upper, gamma_p_T : all arrays

    print("gamma and N:", np.sum(has_H2_and_He_gamma_N), np.sum(has_He_but_not_H2_gamma_N), np.sum(has_only_air_gamma_N))
    print("deltas:", np.sum(has_H2_and_He_delta), np.sum(has_He_but_not_H2_delta), np.sum(has_H2_but_not_He_delta), np.sum(has_air_but_not_H2_and_He_delta), np.sum(has_no_delta))
    
    ##################
    counter = 0
    for P_index, P in enumerate(P_grid):
        P_atm = P / BAR_TO_PASCAL
        cutoff = max(25 * P_atm, 100)
        if P_atm <= 1:
            cutoff = 25
        else:
            cutoff = min(25*P_atm, 100)
        
        lower_indexes = np.searchsorted(lines_array[:,0], wavenums - cutoff, side='right') 
        upper_indexes = np.searchsorted(lines_array[:,0], wavenums + cutoff)
        for T_index, T in enumerate(T_grid):
            print("Currently processing logP,T", np.log10(P), T)
            gamma_p_T = np.zeros(len(a))
            gamma_p_T[has_H2_and_He_gamma_N] = 0.85 * P_atm * (T_ref/ T)**(n_H2[has_H2_and_He_gamma_N]) * gamma_H2[has_H2_and_He_gamma_N] \
                                               + 0.15 * P_atm * (T_ref / T)**(n_He[has_H2_and_He_gamma_N]) * gamma_He[has_H2_and_He_gamma_N]
            gamma_p_T[has_He_but_not_H2_gamma_N] = P_atm * (T_ref / T)**(n_He[has_He_but_not_H2_gamma_N]) * gamma_He[has_He_but_not_H2_gamma_N]
            gamma_p_T[has_H2_but_not_He_gamma_N] = P_atm * (T_ref / T)**(n_H2[has_H2_but_not_He_gamma_N]) * gamma_H2[has_H2_but_not_He_gamma_N]
            gamma_p_T[has_only_air_gamma_N] = P_atm * (T_ref / T)**(n_air[has_only_air_gamma_N]) * gamma_air[has_only_air_gamma_N]

            v_ij_star = np.copy(v_ij)
            v_ij_star[has_H2_and_He_delta] += P_atm * (delta_H2[has_H2_and_He_delta] * 0.85 + \
                                                                              delta_He[has_H2_and_He_delta] * 0.15)
            v_ij_star[has_He_but_not_H2_delta] += P_atm * delta_He[has_He_but_not_H2_delta]
            v_ij_star[has_H2_but_not_He_delta] += P_atm * delta_H2[has_H2_but_not_He_delta]
            v_ij_star[has_air_but_not_H2_and_He_delta] += P_atm * delta_air[has_air_but_not_H2_and_He_delta]
            absorption_cross_section = np.zeros(len(wavenums))
            for i in range(len(wavenums)):
                #if i % 100000 == 0: 
                #    print("Processing wavenum", i)
                lower = lower_indexes[i]
                upper = upper_indexes[i]
                Q = np.interp(T, Q_temperatures, Qs)

                
                absorption_cross_section[i] = compute_one_wavenum(wavenums[i], T, iso_abundance, iso_mass, Q, \
                                                                  v_ij_star[lower : upper], a[lower : upper], \
                                                                  elower[lower : upper], g_upper[lower : upper], \
                                                                  gamma_p_T[lower : upper])
                if i == 4000 and P==1e4 and T==1000:
                    num_lines = upper - lower + 1
                    print("at 4000 I got", absorption_cross_section[4000])


            #absorption_cross_section = absorption_cross_section[::-1]
            if T_index == 9 and P_index == 8:
                print("Before sending, I get {} for 3616".format(absorption_cross_section[3616]))
            comm.send(absorption_cross_section, dest=0, tag=counter)
            counter += 1

          

if rank == 0:
    absorption = handle_master("(12C)(16O)")
    np.save("absorption_parallel.npy", absorption)
else:
    handle_slave()


