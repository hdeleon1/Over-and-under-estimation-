from __future__ import print_function
import math
import multiprocessing as mp
import ctypes
import numpy as np
from random import seed
from random import random
import scipy.io
from scipy.io import savemat
from scipy.io import loadmat
import time
import functools



""""
def make_prob(M_not_ill_x1,M_not_ill_y1,M_ill_x1,M_ill_y1,n,Who_not_ill_matrix,Who_ill_matrix,Who_ill2_matrix):
    T_x = np.abs(M_not_ill_x1 - M_ill_x1)
    T_y = np.abs(M_not_ill_y1 - M_ill_y1)
    T = pow(T_x, 2) + pow(T_y, 2)
    T_R = np.exp(-T / 2 / 2.4)

    T2 = np.power(Who_not_ill_matrix - Who_ill_matrix, 2)

    r_t_a = np.exp(-pow(n - Who_ill2_matrix - 4, 2) / 2)

    r_t_b = (np.exp(-pow(n - Who_ill2_matrix - 4, 2) / 2) +
             np.exp(-pow(n - Who_ill2_matrix - 5, 2) / 2) +
             np.exp(-pow(n - Who_ill2_matrix - 6, 2) / 2) +
             np.exp(-pow(n - Who_ill2_matrix - 7, 2) / 2))

    K_a = np.exp(-T2 / 2 / pow(3, 2)) / 2
    K_b = np.exp(-T2 / 2 / pow(5, 2)) + \
          np.exp(-T2 / 2 / pow(125, 2)) / 5
    P_a[i * n1:n1 + i * n1, 0] = np.sum(np.multiply(r_t_a, K_a), axis=0)
    P_b[i * n1:n1 + i * n1, 0] = np.sum(np.multiply(r_t_a, K_b, T_R), axis=0)
    return P_a,P_b
"""



def shared_array_first(shape):
    shared_array_base = mp.Array(ctypes.c_double, shape[0] * shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array


    # Form a shared array and a lock, to protect access to shared memory.

    #lock = multiprocessing.Lock()


#@functools.lru_cache(maxsize=128)
#def parallel_function(l,def_param=(temp_array_Who_ill1,temp_array_Who_ill2)):
#def parallel_function_all(N):
N= 100 * math.pow(10, 4)


#EE=loadmat('Temp_time_vac_a')
Array_Who_ill1 = np.zeros((int(N),8,10))
Array_Who_ill2 = np.zeros((int(N),8,10))

Array_Who_vac = np.zeros((int(N),20))

data = loadmat('R1.mat')
R1 = data['R1']

data1=loadmat('t_rand.mat')

Who_vac = data1['t_rand']



for l in range(0,10):
    print(l)
    for kk in range(0,10):
        who_vac = Who_vac[:, kk]
        print(who_vac.shape)
        Who_iil_day = np.zeros((250, 1400, 10))
        temp_array_Who_ill1 = np.zeros((int(N)))
        temp_array_Who_ill2 = np.zeros((int(N)))

        seed(time)

        v1 = np.arange(0, N, 1)
        N_start = int(350)
        N_ill = np.floor(N * np.random.rand(N_start))

        which_day=np.zeros(350)

        v1[N_ill.astype(int)] = 0
        Who_not_ill = v1[v1 > 0]

        M_x = np.arange(0, N, 1)
        M_y = np.arange(0, N, 1)

        Who_ill1 = np.zeros(N_start)
        Who_ill2 = np.zeros(N_start)

        M_ill_x0 = M_ill_y0 = np.zeros(N_start)

        Who_ill1[0:len(N_ill)] = N_ill

        rand_ill=Who_ill1[0:len(N_ill)]

        n_ill = len(N_ill)

        k = 5
        R_end = (10 - k * 3 / 20) * math.pow(10, 3)/10
        r_end = R_end

        M_ill_x0[0:n_ill] = R_end * M_x[Who_ill1.astype(int)] / N
        M_ill_y0[0:n_ill] = R_end * M_y[Who_ill1.astype(int)] / N

        M_ill_x = M_ill_x0
        M_ill_y = M_ill_y0

        # Who_not_ill=setdiff(1:N,Who_ill(:,1));
        M_not_ill_x = R_end * M_x[v1 > 0] / N
        M_not_ill_y = R_end * M_y[v1 > 0] / N
        start_time = time.time()


        seed(time)
        who_vac_1 = who_vac

        #scipy.io.savemat('who',{'who':who_vac_1})


        for n in range(0, 150):
            new_ill=np.zeros((1))
            k = 69 * pow(1.15, 1.85) - 85

            if n<55:
                k = 69 * pow(R1[n+100], 1.85) - 85

            R_end = 1.7*(10 - k * 5 / 20) * math.pow(10, 3)/10
            r_end = R_end
        #print(n, "k=",k,"r_end=",r_end)

        # print(r_end)
        #rr=0.99

            rr=0.2



            seed(time)
            if np.random.rand(1) > rr:
                #print(n,rr)
                if n<160:
                    a = np.floor(np.random.rand(20) * len(Who_not_ill))
                a1 = Who_not_ill[a.astype(int)]
                who =who_vac_1[a1.astype(int)]




                left_time = n * np.ones(len(who)) - who
                # print(np.min(who))
                # print(np.min(who_vac_1))

                p = np.zeros(len(a1))

                p =0.85 * (16.5 - np.exp(-left_time * 0.2)) / 16.5  # vaccine increases as s function of time
                p[left_time <= -14] = 0
                p=np.ones(20)-p
                #print(np.sum(p))
                p=np.floor(np.random.rand(20)+ p )
                #print(np.sum(p))
                A = Who_not_ill[a.astype(int)]
                A=A[p==1]
                a=a[p==1]
                print(n,len(A))
                b = np.ones(len(A)) * (n + 1)

                Who_ill1 = np.concatenate((Who_ill1, A))
                rand_ill = np.concatenate((rand_ill, A))
                which_day = np.concatenate((which_day,b))
                if n%10==0:
                    scipy.io.savemat('N_ill.mat', {'N_ill': rand_ill,'which_day':which_day})

                Who_ill2 = np.concatenate((Who_ill2, b))
                #print(len(M_not_ill_x),np.max(a))
                M_ill_x = np.concatenate((M_ill_x, M_not_ill_x[a.astype(int)]))
                M_ill_y = np.concatenate((M_ill_y, M_not_ill_y[a.astype(int)]))
                M_not_ill_x[a.astype(int)] = 0
                M_not_ill_y[a.astype(int)] = 0
                M_not_ill_x = M_not_ill_x[M_not_ill_x > 0]
                M_not_ill_y = M_not_ill_y[M_not_ill_y > 0]
                Who_not_ill[a.astype(int)] = 0
                Who_not_ill = Who_not_ill[Who_not_ill > 0]





            n_not_ill = len(Who_not_ill)
            n_ill = len(Who_ill1)
            #print(n_not_ill)
            # print (len(Who_ill2))

            l_x = np.random.normal(2, 1, n_not_ill)
            l_y = np.random.normal(2, 1, n_not_ill)

            delta_x1_not_ill = pow(10, l_x)
            delta_y1_not_ill = pow(10, l_y)

            l_x = np.random.normal(2, 1, n_ill)
            l_y = np.random.normal(2, 1, n_ill)

            delta_x1_ill = pow(10, l_x)
            delta_y1_ill = pow(10, l_y)

            M_not_ill_x = M_not_ill_x + delta_x1_not_ill
            M_not_ill_y = M_not_ill_y + delta_y1_not_ill
            M_not_ill_x = np.remainder(M_not_ill_x, r_end)
            M_not_ill_y = np.remainder(M_not_ill_y, r_end)

            M_ill_x = M_ill_x + delta_x1_ill
            M_ill_y = M_ill_y + delta_y1_ill
            M_ill_x = np.remainder(M_ill_x, r_end)
            M_ill_y = np.remainder(M_ill_y, r_end)


            def shared_array(shape):
                shared_array_base = mp.Array(ctypes.c_double, shape[0] * shape[1])
                shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
                shared_array = shared_array.reshape(*shape)
                return shared_array


            P_a = shared_array((int(n_not_ill),1 ))
            P_b = shared_array((int(n_not_ill),1 ))

            Y = np.argwhere((n - Who_ill2) < 9)
            #print(n, len(Who_ill2), len(Y))
            # print(len(Y))
            who_ill1 = Who_ill1[Y]
            who_ill2 = Who_ill2[Y]
            m_ill_x = M_ill_x[Y]
            m_ill_y = M_ill_y[Y]

            n_o_p = 60


            # print (n_not_ill%n_o_p)




            #print(who_ill1)
            def parallel_function_T_r(i, def_param=(
                    m_ill_x, m_ill_y, M_not_ill_x, Who_not_ill, who_ill1, who_ill2, M_not_ill_y, n_not_ill, n_o_p)):
                n1 = math.floor(n_not_ill / n_o_p)
                n_ill = len(who_ill2)
                #print(n1)
                for j in range(n1*i, n1*(i+1)):
                    #print(j)

                    T_x =np.power(M_not_ill_x[j] - m_ill_x,2)
                    T_y =np.power(M_not_ill_y[j] - m_ill_y,2)
                    T=T_x+T_y
                    T[T<9]=9
                    #print(j)


                    T_R = np.exp(-T / 2 / pow(2.4,2))

                    R_t= n - who_ill2
                    N_dis=np.abs(Who_not_ill[j]-who_ill1)
                    #N_dis[N_dis>N/2]=N/2-np.remainder(N_dis[N_dis>N/2],N/2)
                    #print(N_dis)


                    T2=np.power(N_dis,2)#Periodic boundary conditions dor the population.

                #R_t=R_t[KK[:,0],KK[:,1]]
                #print(R_t.shape)



                    r_t_a = np.exp(-pow(R_t - 4, 2) / 2)

                    r_t_b = (np.exp(-pow(R_t - 4, 2) / 2) +
                         np.exp(-pow(R_t - 5, 2) / 2) +
                         np.exp(-pow(R_t - 6, 2) / 2) +
                         np.exp(-pow(R_t - 7, 2) / 2))

                    K_a = np.exp(-T2 / 2 / pow(3, 2)) / 2
                    K_b = np.exp(-T2 / 2 / pow(25, 2)) + \
                          np.exp(-T2 / 2 / pow(100, 2)) / 5
                    p_a = np.sum(0.1 * (r_t_a*K_a))
                    p_b = np.sum(r_t_b * T_R * K_b)

                    P_a[j,0] = p_a
                    P_b[j,0] = p_b









            pool = mp.Pool(processes=n_o_p)
            args = range(0, n_o_p)
            pool.map(parallel_function_T_r, args)
            pool.close()
            pool.join()

            #print(n_not_ill)

            """"
            #print( n_o_p % n_not_ill)
            if (n_o_p % n_not_ill  > 0):
    
                n1 = math.floor(n_not_ill / n_o_p)
                n2 = n_not_ill % n_o_p
                M_ill_x1 = M_ill_x.reshape(-1, 1) * np.ones((1, n2))
                M_ill_y1 = M_ill_y.reshape(-1, 1) * np.ones((1, n2))
    
                M_not_ill_x1 = np.ones((n_ill, 1)) * M_not_ill_x[n_o_p * n1:n2 + n_o_p * n1]
                M_not_ill_y1 = np.ones((n_ill, 1)) * M_not_ill_y[n_o_p * n1:n2 + n_o_p * n1]
    
                Who_ill_matrix = Who_ill1.reshape(-1, 1) * np.ones((1, n2))
                Who_ill2_matrix = Who_ill2.reshape(-1, 1) * np.ones((1, n2))
                Who_not_ill_matrix = np.ones((n_ill, 1)) * Who_not_ill[n_o_p * n1:n2 + n_o_p * n1]
                
    
                # [p_a,p_b]=make_prob(M_not_ill_x1,M_not_ill_y1,M_ill_x1,M_ill_y1,n,Who_not_ill_matrix,Who_ill_matrix,Who_ill2_matrix)
    
                M_ill_x1 = M_ill_x.reshape(-1, 1) * np.ones((1, n_not_ill))
                M_ill_y1 = M_ill_y.reshape(-1, 1) * np.ones((1, n_not_ill))
    
                M_not_ill_x1 = np.ones((n_ill, 1)) * M_not_ill_x
                M_not_ill_y1 = np.ones((n_ill, 1)) * M_not_ill_y
    
                Who_ill_matrix = Who_ill1.reshape(-1, 1) * np.ones((1, n_not_ill))
                Who_ill2_matrix = Who_ill2.reshape(-1, 1) * np.ones((1, n_not_ill))
                Who_not_ill_matrix = np.ones((n_ill, 1)) * Who_not_ill
                
                T_x = np.abs(M_not_ill_x1 - M_ill_x1)
                T_y = np.abs(M_not_ill_y1 - M_ill_y1)
                T = pow(T_x, 2) + pow(T_y, 2)
                T[T<8]=8
                T_R = np.exp(-T / 2 / pow(2.4,2))
    
                N_dis = np.abs(Who_not_ill_matrix - Who_ill_matrix)
                #N_dis[N_dis > N / 2] = N / 2 - np.remainder(N_dis[N_dis > N / 2], N / 2)
                if n==0:
                    scipy.io.savemat('N_dis.mat', {'N_dis': N_dis,'new_ill':new_ill})
                T2 = np.power(N_dis, 2)
    
                r_t_a = np.exp(-pow(n - Who_ill2_matrix - 4, 2) / 2)
    
                r_t_b = (np.exp(-pow(n - Who_ill2_matrix - 4, 2) / 2) +
                         np.exp(-pow(n - Who_ill2_matrix - 5, 2) / 2) +
                         np.exp(-pow(n - Who_ill2_matrix - 6, 2) / 2) +
                         np.exp(-pow(n - Who_ill2_matrix - 7, 2) / 2))
    
                K_a = np.exp(-T2 / 2 / pow(3, 2)) / 2
                K_b =np.exp(-T2 / 2 / pow(25, 2)) + \
                      np.exp(-T2 / 2 / pow(75, 2)) / 5
                p_a = np.sum(0.1 * (r_t_a*K_a), axis=0)
                p_b = np.sum(r_t_b*T_R*K_b, axis=0)
    
                P_a[n_o_p * n1:n2 + n_o_p * n1,0] = p_a
                P_b[n_o_p * n1:n2 + n_o_p * n1,0] = p_b
                #scipy.io.savemat('P', {'P_a': P_a, 'P_b': P_b})
                # print (np.shape(P_b),np.shape(P_a))
                
                
            """




            who=who_vac_1[Who_not_ill.astype(int)]
            left_time=n*np.ones(len(who))-who
            #print(np.min(who))
            #print(np.min(who_vac_1))fuc

            p = np.zeros(len(Who_not_ill))


            p = 0.85*(16.5-np.exp(-left_time*0.2))/16.5 #vaccine increases as a function of time
            p[left_time <= -14] = 0
            #print('sum p=', np.sum(p))
            P_b[P_b>2*2/100]=2*2/100

            P = np.random.rand(len(P_b)) + 2.2* ((P_b[:,0]*5  + 1*P_a[:,0])*(np.ones(len(p)) - p))

            P1= np.floor(P)
            #print(new_ill)
            if n%10:
                scipy.io.savemat('N_dis.mat', {'new_ill': Who_ill1,'Who_ill2':Who_ill2,'P':P,'P_a':P_a,'P_b':P_b,'p':p})

            #if n%5==0:
             #   scipy.io.savemat('P1.mat',
              #                   {'p': p, 'who': who, 'left_time': left_time,'P1':P1,'P_a':P_a,'P_b':P_b,'Who_not_ill':Who_not_ill})



            if np.sum(P1) > 0:
                print(n,sum(P1))
                a1 = np.argwhere(P > 1)
                a1 = a1[:, 0]
                b1 = np.ones(len(a1)) * (n + 1)
                #Who_vac1[n,0:len(who[a1.astype(int)]),l]=who[a1.astype(int)]
                new_ill = np.concatenate((new_ill, Who_not_ill[a1.astype(int)]))


                # b[0] = n
                # print(a1.astype(int))

                Who_ill1 = np.concatenate((Who_ill1, Who_not_ill[a1.astype(int)]))
                Who_ill2 = np.concatenate((Who_ill2, b1.astype(int)))

                # Who_ill2 = np.concatenate((Who_ill2, b))
                M_ill_x = np.concatenate((M_ill_x, M_not_ill_x[a1.astype(int)]))
                M_ill_y = np.concatenate((M_ill_y, M_not_ill_y[a1.astype(int)]))
                M_not_ill_x[a1.astype(int)] = 0
                M_not_ill_y[a1.astype(int)] = 0
                M_not_ill_x = M_not_ill_x[M_not_ill_x > 0]
                M_not_ill_y = M_not_ill_y[M_not_ill_y > 0]
                Who_not_ill[a1.astype(int)] = 0
                Who_not_ill = Who_not_ill[Who_not_ill > 0]
                # print (len(Who_ill1))
                # print (len(Who_not_ill))
                #

            Who_iil_day[n, 0:len(new_ill), l] = new_ill
            scipy.io.savemat('who_d.mat', {'who': who_vac,
                                           'who_ill': Who_iil_day})
            #print(new_ill)
            #print(Who_iil_day[0, :])
            n_ill=len(Who_ill2)
            temp_array_Who_ill2[1:n_ill] = Who_ill2[1:n_ill]
            temp_array_Who_ill1[1:n_ill] = Who_ill1[1:n_ill]
            #print(who_vac_1[100])
          #  scipy.io.savemat('temp_vac_time.mat', {'temp1': temp_array_Who_ill1, 'temp2': temp_array_Who_ill2,'who_vac1':who_vac_1,'who_not_ill':Who_not_ill})


        print(time.time() - start_time)
        Array_Who_ill1[0:len(temp_array_Who_ill1), kk, l] = temp_array_Who_ill1
        Array_Who_ill2[0:len(temp_array_Who_ill2), kk, l] = temp_array_Who_ill2
    #Array_Who_vac[:,l] = who_vac_1
        scipy.io.savemat('Temp_time_vac_rand_a.mat', {'temp1': Array_Who_ill1, 'temp2': Array_Who_ill2,'who_vac':who_vac_1})
    # return temp_array_Who_ill1,temp_array_Who_ill2


""""
if __name__ == '__main__':


    #pool = mp.Pool(processes=1)

    # Call the parallel function with different inputs.
    #args = range(0, 1)

    # Use map - blocks until all processes are done.
    #pool.map(parallel_function_all, args)
N = 100 * math.pow(10, 4)
[temp_array_Who_ill1,temp_array_Who_ill2]=parallel_function_all(N)
"""